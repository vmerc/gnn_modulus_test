import time
import hydra
import torch
import sys
import os
import torch.nn as nn
import argparse
import cupy as cp
import numpy as np
from dgl.dataloading import GraphDataLoader
import dgl

project_path = os.path.abspath(os.path.join(os.getcwd(), '..', ''))
sys.path.append(project_path)

from python.TestRollout import TestRollout

from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_wandb,
)

from python.python_code.data_manip.extraction.telemac_file import TelemacFile
from python.create_dgl_dataset import add_mesh_info

import pickle
from scipy.interpolate import griddata
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay

import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

def hann_2d(Nx, Ny, dtype=np.float32):
    """Fenêtre de Hann 2D en float32."""
    w_x = np.hanning(Nx).astype(dtype)
    w_y = np.hanning(Ny).astype(dtype)
    return np.outer(w_x, w_y)

def radial_average_spectrum_cupy(spectrum_gpu, center=None):
    ny, nx = spectrum_gpu.shape
    if center is None:
        center = (nx//2, ny//2)
    x0, y0 = center

    y_gpu, x_gpu = cp.indices((ny, nx))
    r_gpu = cp.sqrt((x_gpu - x0)**2 + (y_gpu - y0)**2)
    r_int_gpu = r_gpu.astype(cp.int32)

    r_int_flat_gpu = r_int_gpu.ravel()
    spectrum_flat_gpu = spectrum_gpu.ravel()
    
    r_max = int(r_int_gpu.max().get())

    sumvals_gpu = cp.bincount(r_int_flat_gpu, weights=spectrum_flat_gpu)
    counts_gpu  = cp.bincount(r_int_flat_gpu)
    sumvals_gpu  = sumvals_gpu.astype(cp.float32)
    counts_gpu   = counts_gpu.astype(cp.float32)
    rad_ave_gpu = sumvals_gpu / cp.where(counts_gpu == 0, cp.float32(1), counts_gpu)

    r_vals_gpu = cp.arange(r_max+1, dtype=cp.float32)

    r_vals_cpu = cp.asnumpy(r_vals_gpu)
    rad_ave_cpu = cp.asnumpy(rad_ave_gpu)
    return r_vals_cpu, rad_ave_cpu

def radial_average_spectrum_cupy_physical(
    spectrum_gpu,
    dx, dy,
    center=None,
    k_step=None
):
    ny, nx = spectrum_gpu.shape
    if center is None:
        cx = nx // 2
        cy = ny // 2
    else:
        cx, cy = center

    # Build the wavenumber grid
    y_idx_gpu, x_idx_gpu = cp.indices((ny, nx))
    kx_gpu = (x_idx_gpu - cx) * (2.0 * np.pi / (nx * dx))
    ky_gpu = (y_idx_gpu - cy) * (2.0 * np.pi / (ny * dy))
    r_k_gpu = cp.sqrt(kx_gpu**2 + ky_gpu**2)

    # Flatten
    r_k_flat_gpu = r_k_gpu.ravel()
    spectrum_flat_gpu = spectrum_gpu.ravel()

    del x_idx_gpu, y_idx_gpu, kx_gpu, ky_gpu, r_k_gpu
    cp._default_memory_pool.free_all_blocks()

    # -------------------------------------------------------------------
    # 1) We REQUIRE a user-supplied k_step here (the same for all runs).
    # 2) k_max depends on domain & spacing => "theoretical Nyquist"
    #    e.g., sqrt((2π/dx)^2 + (2π/dy)^2).
    # -------------------------------------------------------------------
    if k_step is None:
        raise ValueError("You must pass a 'k_step' so all runs share the same bin width!")

    # Example formula for max radial wavenumber:
    k_max = np.sqrt( (2.0*np.pi/dx)**2 + (2.0*np.pi/dy)**2 )

    # Number of bins for THIS run
    nbins = int(np.floor(k_max / k_step)) + 1  # +1 to include top edge
    # If you want a margin above the strict Nyquist, you can add a factor

    # Convert each radius to bin index
    bin_idx_gpu = cp.floor(r_k_flat_gpu / k_step).astype(cp.int32)

    # Clip any indices that exceed nbins-1
    bin_idx_gpu = cp.where(bin_idx_gpu >= nbins, nbins - 1, bin_idx_gpu)

    # Bin counts and sums
    sumvals_gpu = cp.bincount(bin_idx_gpu, weights=spectrum_flat_gpu, minlength=nbins)
    counts_gpu  = cp.bincount(bin_idx_gpu, minlength=nbins)

    # Avoid div-by-zero
    rad_ave_gpu = sumvals_gpu / cp.where(counts_gpu == 0, cp.float32(1), counts_gpu)

    # Define bin centers => (i + 0.5)*k_step
    k_vals_gpu = (cp.arange(nbins, dtype=cp.float32) + 0.5) * k_step

    # Move to CPU
    k_vals_cpu = cp.asnumpy(k_vals_gpu)
    rad_ave_cpu = cp.asnumpy(rad_ave_gpu)

    return k_vals_cpu, rad_ave_cpu


def process_fft_and_radial(data_cpu, window_2d_cpu,global_k_step):
    """
    data_cpu  : un tableau NumPy (Ny, Nx, nb_channels) en float32 (de préférence).
    window_2d_cpu : fenêtre Hann (Ny, Nx) en float32.

    On:
    1) traite canal par canal
    2) fait la fft2 sur GPU
    3) calcule la moyenne radiale
    4) stocke le profil rad (sur CPU) dans un tableau de résultat
    Retourne: 
      - r_vals (un seul vecteur rayon, identique pour tous les canaux si la taille ne change pas)
      - rad_profiles : shape (nb_channels, len(r_vals))
    """
    Ny, Nx, nchan = data_cpu.shape
    # On copie la fenêtre sur GPU une seule fois
    window_2d_gpu = cp.asarray(window_2d_cpu, dtype=cp.float32)

    rad_profiles = []
    r_vals_common = None

    for c in range(nchan):
        # Copie du canal c sur GPU
        slice_cpu = data_cpu[..., c]  # shape (Ny, Nx)
        slice_gpu = cp.asarray(slice_cpu, dtype=cp.float32)

        # Fenêtrage (multiplication par Hann)
        slice_gpu *= window_2d_gpu

        # FFT
        slice_fft_gpu = cp.fft.fft2(slice_gpu)
        slice_fft_shifted_gpu = cp.fft.fftshift(slice_fft_gpu)

        # |FFT| and normalization
        slice_mag_gpu = cp.abs(slice_fft_shifted_gpu) / (Ny * Nx)

        # Calcul moyenne radiale
        r_vals, rad_ave = radial_average_spectrum_cupy_physical(slice_mag_gpu,
                                                                dx=min_length,       
                                                                dy=min_length,       
                                                                center=(Nx//2, Ny//2),
                                                                k_step=global_k_step          
                                                                )
        if r_vals_common is None:
            r_vals_common = r_vals
        rad_profiles.append(rad_ave)

        # Libération mémoire GPU temporaire
        del slice_gpu, slice_fft_gpu, slice_fft_shifted_gpu, slice_mag_gpu
        cp._default_memory_pool.free_all_blocks()

    rad_profiles = np.array(rad_profiles)  # shape (nchan, len(r_vals))
    return r_vals_common, rad_profiles

def global_smallest_edge_length_vectorized(X, triangles):
    """
    Trouve la longueur minimale globale des edges parmi tous les triangles de manière vectorisée.

    Parameters:
        X (numpy.ndarray): Un tableau de forme (n, 2) ou (n, 3), contenant les coordonnées des sommets.
        triangles (numpy.ndarray): Un tableau de forme (m, 3), contenant les indices des sommets de chaque triangle.

    Returns:
        float: La longueur minimale globale des edges.
    """
    # Récupérer les coordonnées des sommets pour chaque triangle
    p1 = X[triangles[:, 0]]
    p2 = X[triangles[:, 1]]
    p3 = X[triangles[:, 2]]
    # Calculer les longueurs des edges
    edge_lengths = np.array([
        np.linalg.norm(p1 - p2, axis=1),  # Longueur entre p1 et p2
        np.linalg.norm(p2 - p3, axis=1),  # Longueur entre p2 et p3
        np.linalg.norm(p3 - p1, axis=1)   # Longueur entre p3 et p1
    ])
    print(edge_lengths.shape)
    # Trouver le minimum global
    min_length = np.min(edge_lengths)
    
    return min_length

def interpolate_values_cupy(
    tri_simplices,
    simplex_indices,
    bary_coords,
    values,
    valid_indices,
    n_grid_points
):
    """
    Version vectorisée sur GPU de votre fonction d'interpolation.
    """
    # 1) On copie les gros tableaux en CuPy
    tri_simplices_gpu    = cp.asarray(tri_simplices)      # (n_simplices, 3)
    simplex_indices_gpu  = cp.asarray(simplex_indices)    # (n_grid_points,)
    bary_coords_gpu      = cp.asarray(bary_coords)        # (n_valid, 3)
    values_gpu           = cp.asarray(values)             # (n_nodes, n_vars)
    valid_indices_gpu    = cp.asarray(valid_indices)      # (n_valid,)

    n_vars = values.shape[1]

    # Crée le tableau résultat (sur GPU) : (n_grid_points, n_vars)
    result_gpu = cp.zeros((n_grid_points, n_vars), dtype=cp.float32)

    # Indices de simplex effectifs (pour les points valides)
    chosen_simplex = simplex_indices_gpu[valid_indices_gpu]  # (n_valid,)

    # On récupère les 3 nœuds de chaque simplex
    # tri_simplices_gpu[chosen_simplex].shape = (n_valid, 3)
    tri_pts = tri_simplices_gpu[chosen_simplex]  # indices de sommets

    # On « déplie » pour récupérer les valeurs associées aux sommets
    # tri_pts.reshape(-1) a shape (n_valid*3,)
    # values_gpu[ ... ] aura shape (n_valid*3, n_vars)
    pred_flat = values_gpu[tri_pts.reshape(-1)]  
    # On redimensionne pour retrouver l'axe des 3 sommets
    # => (n_valid, 3, n_vars)
    pred_flat = pred_flat.reshape(-1, 3, n_vars)

    # Barycentres : (n_valid, 3) -> (n_valid, 3, 1)
    bary_coords_expanded = bary_coords_gpu[:, :, None]  # Ajout d'un axe singleton

    # Produit barycentrique vectorisé :
    #   (n_valid, 3, n_vars) * (n_valid, 3, 1)
    # puis somme sur l’axe=1 (les 3 sommets)
    interpolated_val = (pred_flat * bary_coords_expanded).sum(axis=1)  # (n_valid, n_vars)

    # On range le résultat aux bonnes positions
    result_gpu[valid_indices_gpu] = interpolated_val

    # On rapatrie sur CPU
    result_cpu = cp.asnumpy(result_gpu)

    return result_cpu

def interpolate_values_cupy_in_chunks(
    tri_simplices, 
    simplex_indices, 
    bary_coords, 
    values, 
    valid_indices, 
    n_grid_points,
    chunk_size=200_000  # adjust chunk size if still OOM
):
    # Transfer large, read-only arrays to GPU once
    tri_simplices_gpu   = cp.asarray(tri_simplices) 
    simplex_indices_gpu = cp.asarray(simplex_indices)
    values_gpu          = cp.asarray(values)
    valid_indices_gpu   = cp.asarray(valid_indices)
    bary_coords_gpu     = cp.asarray(bary_coords)

    n_vars = values.shape[1]  # e.g., 3 dynamic features (h, u, v)
    result_gpu = cp.zeros((n_grid_points, n_vars), dtype=cp.float32)

    # Identify which simplex each valid point belongs to
    chosen_simplex_gpu = simplex_indices_gpu[valid_indices_gpu]

    n_valid = valid_indices_gpu.shape[0]
    num_chunks = (n_valid + chunk_size - 1) // chunk_size
    print(f"Interpolating {n_valid} points in {num_chunks} chunks of size up to {chunk_size}.")

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i+1) * chunk_size, n_valid)

        # Subset of valid indices
        valid_sub_gpu   = valid_indices_gpu[start:end]       # shape: (sub_chunk,)
        chosen_sub_gpu  = chosen_simplex_gpu[start:end]      # shape: (sub_chunk,)
        bary_sub_gpu    = bary_coords_gpu[start:end]         # shape: (sub_chunk, 3)

        # Get the triangle node indices for these points
        tri_pts_sub_gpu = tri_simplices_gpu[chosen_sub_gpu]  # shape: (sub_chunk, 3)

        # Gather node values -> (sub_chunk * 3, n_vars)
        pred_flat_sub_gpu = values_gpu[tri_pts_sub_gpu.reshape(-1)]

        # Reshape -> (sub_chunk, 3, n_vars)
        pred_flat_sub_gpu = pred_flat_sub_gpu.reshape(-1, 3, n_vars)

        # Expand bary coords -> (sub_chunk, 3, 1)
        bary_sub_expanded_gpu = bary_sub_gpu[:, :, None]

        # Multiply and sum over the '3' axis
        # => shape: (sub_chunk, n_vars)
        interpolated_sub_gpu = (pred_flat_sub_gpu * bary_sub_expanded_gpu).sum(axis=1)

        # Store result back
        result_gpu[valid_sub_gpu] = interpolated_sub_gpu

        # Free chunk memory
        del (valid_sub_gpu, chosen_sub_gpu, bary_sub_gpu,
             tri_pts_sub_gpu, pred_flat_sub_gpu,
             bary_sub_expanded_gpu, interpolated_sub_gpu)
        cp._default_memory_pool.free_all_blocks()
    # Transfer final array back to CPU
    return cp.asnumpy(result_gpu)




parser = argparse.ArgumentParser(description="Script to handle configuration, saving, and mesh processing.")
    
# Add arguments
parser.add_argument("--config_name", type=str, required=True, help="Name of the configuration file (string).")
parser.add_argument("--saving_name", type=str, required=True, help="Name for the saving file (string).")
parser.add_argument("--mesh_list", type=str, required=True, help="List of meshes as a string.")
parser.add_argument("--load", action="store_true", help="Boolean flag to indicate whether to load (default: False).")
    
# Parse arguments
args = parser.parse_args()
    
# Access the arguments
config_name = args.config_name
saving_name = args.saving_name
mesh_list = [args.mesh_list]
load = args.load

# Initialize Hydra and set the configuration directory
with initialize(config_path="../bin/conf"):
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()
    # Compose the configuration using the config name
    cfg = compose(config_name=config_name)
    
    # Now call the training function with the composed config
    test = TestRollout(cfg,logger)
    
    predict,groundtruth,origin = test.predict_unroll(unroll_steps=12)
    
#del test 
torch.cuda.empty_cache()

res_mesh = TelemacFile(mesh_list[0])
X,triangles = add_mesh_info(res_mesh)

min_length = global_smallest_edge_length_vectorized(X,triangles)

res_mesh_fine = TelemacFile('/work/m24046/m24046mrcr/results_data_30min/maillage_3.slf')
X_fine,triangles_fine = add_mesh_info(res_mesh_fine)

fine_values = []
list_of_test_files = [
    '/work/m24046/m24046mrcr/results_data_30min_35_70/Group_3_peak_1600_Group_3_peak_1600_0_0-80.pkl',
    '/work/m24046/m24046mrcr/results_data_30min_35_70/Group_3_peak_2200_Group_3_peak_2200_0_0-80.pkl',
    '/work/m24046/m24046mrcr/results_data_30min_35_70/Group_3_peak_3200_Group_3_peak_3200_0_0-80.pkl'
]

sequence_length = 13
overlap = 13
sequences = []
for file_path in list_of_test_files:
    with open(file_path, 'rb') as f:
        dynamic_data = pickle.load(f)
        step = max(1, sequence_length - overlap)
        for i in range(0, len(dynamic_data) - sequence_length + 1, step):
            sequence = dynamic_data[i:i + sequence_length]
            sequences.append(sequence)
            
fine_gd = []
fine_ori = []
for seq in sequences : 
    a,b = seq[-1]
    correct = a+b
    fine_gd.append(np.float32(correct))
    fine_ori.append(np.float32(seq[0][0]))
    
# Définir les bornes de votre domaine
xmin = min(X[:, 0].min(), X_fine[:, 0].min())
xmax = max(X[:, 0].max(), X_fine[:, 0].max())
ymin = min(X[:, 1].min(), X_fine[:, 1].min())
ymax = max(X[:, 1].max(), X_fine[:, 1].max())

# Calculer le nombre de points dans chaque dimension en fonction de min_length
Nx = int(np.ceil((xmax - xmin) / min_length)) + 1
Ny = int(np.ceil((ymax - ymin) / min_length)) + 1

# Créer une grille régulière avec min_length comme écart
grid_x, grid_y = np.mgrid[xmin:xmax:Nx*1j, ymin:ymax:Ny*1j]
grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T  # Shape: (n_grid_points, 2)

# Create a mask for the domain using the fine mesh triangulation
triangulation_mask = Triangulation(X_fine[:, 0], X_fine[:, 1], triangles=triangles_fine)
mask = triangulation_mask.get_trifinder()(grid_x, grid_y) != -1  # True inside the domain
mask_flat = mask.ravel()

# Precompute triangulations
tri_coarse = Delaunay(X)
# Compute simplex indices 
def find_simplex_chunk(tri_coarse, points_chunk):
    """Helper function to process a subset of points."""
    return tri_coarse.find_simplex(points_chunk)

N = len(grid_points)

if load : 
    simplex_indices = np.load("./datas/simplex_indices.npy")
else : 
    chunk_size = 500000  # or some value that fits well in memory
    num_chunks = (N + chunk_size - 1) // chunk_size
    print("Nombre de chunks : {}".format(num_chunks))
    # Option 1: Serial chunking
    simplex_indices = np.empty(N, dtype=np.int32)
    offset = 0
    for i in range(num_chunks):
        if i%10 == 0 :
            print(f"Chunk {i} / {num_chunks}", flush=True)
        start = i * chunk_size
        end = min((i+1)*chunk_size, N)
        chunk_result = tri_coarse.find_simplex(grid_points[start:end])
        simplex_indices[start:end] = chunk_result
    #np.save("./datas/simplex_indices.npy", np.array(simplex_indices))
    
# Apply the mask to simplex indices
simplex_indices_coarse_masked = np.where(mask_flat, simplex_indices, -1)

# Identify valid points inside the domain and within a simplex
valid_coarse = simplex_indices_coarse_masked >= 0

# Compute barycentric coordinates for valid points on coarse mesh
transform_coarse = tri_coarse.transform[simplex_indices_coarse_masked[valid_coarse], :2]
offset_coarse = tri_coarse.transform[simplex_indices_coarse_masked[valid_coarse], 2]
delta_coarse = grid_points[valid_coarse] - offset_coarse
bary_coords_coarse = np.einsum('ijk,ik->ij', transform_coarse, delta_coarse)
bary_coords_coarse = np.hstack((bary_coords_coarse, 1 - bary_coords_coarse.sum(axis=1, keepdims=True)))



print("Nx and Ny values {} and {}".format(Nx,Ny))
print("Min length {}".format(min_length))

# Number of grid points
n_grid_points = grid_points.shape[0]

all_rad_profiles_h = []
all_rad_profiles_u = []
all_rad_profiles_v = []

r_vals_common = None  # Pour stocker le vecteur r_vals la première fois
window_2d_cpu = hann_2d(Nx, Ny)


bary_coords_coarse = cp.asarray(bary_coords_coarse, dtype=cp.float32)
simplex_indices_coarse_masked = cp.asarray(simplex_indices_coarse_masked, dtype=cp.int32)
simplices = cp.asarray(tri_coarse.simplices, dtype=cp.int32)

global_k_step = 0.002
for i, pred in enumerate(fine_gd):
    # 1) Interpolation sur la grille
    #    ==========================
    interpolated_values = interpolate_values_cupy_in_chunks(
        tri_coarse.simplices,
        simplex_indices_coarse_masked,
        bary_coords_coarse,
        pred,
        np.where(valid_coarse)[0],
        n_grid_points,
        chunk_size=200000  # or smaller if still OOM
    )
    
    assert np.all(interpolated_values[~mask_flat] == 0), "Non-zero values outside convex hull!"

    
    data_cpu = interpolated_values.reshape((grid_x.shape[0], grid_x.shape[1], -1)).astype(np.float32)

    # 2) FFT et moyenne radiale
    #    ======================

    r_vals, rad_profiles = process_fft_and_radial(data_cpu, window_2d_cpu,global_k_step = global_k_step)
    # rad_profiles : shape (n_vars, len(r_vals)) si 'n_vars' = nb_channels

    # Au besoin, on moyenne sur les canaux
    #rad_profiles_mean = rad_profiles.mean(axis=0)
    # On mémorise le r_vals la première fois, si besoin
    if r_vals_common is None:
        r_vals_common = r_vals

    # On stocke uniquement le profil radial (ou ce qui vous intéresse),
    # pas la grille interpolée en entier
    #all_rad_profiles_h.append(rad_profiles[0,:])
    all_rad_profiles_u.append(rad_profiles[1,:])
    #all_rad_profiles_v.append(rad_profiles[2,:])
    
    cp._default_memory_pool.free_all_blocks()
    
#all_rad_profiles_h = np.array(all_rad_profiles_h)  # shape (N, len(r_vals))
all_rad_profiles_u = np.array(all_rad_profiles_u)  # shape (N, len(r_vals))
#all_rad_profiles_v = np.array(all_rad_profiles_v)  # shape (N, len(r_vals))


# Moyenne sur le temps (ou sur N)
#all_rad_profiles_h_time_mean = all_rad_profiles_h.mean(axis=0)
#all_rad_profiles_u_time_mean = all_rad_profiles_u.mean(axis=0)
#all_rad_profiles_v_time_mean = all_rad_profiles_v.mean(axis=0)

#np.save("./datas/rad_proj_full_h_"+saving_name+".npy", np.array(all_rad_profiles_h))
np.save("./datas/rad_proj_full_u_"+saving_name+".npy", np.array(all_rad_profiles_u))
#np.save("./datas/rad_proj_full_v_"+saving_name+".npy", np.array(all_rad_profiles_v))

np.save("./datas/r_proj_"+variables_name[variable]+"_"+saving_name+".npy", np.array(r_vals))
