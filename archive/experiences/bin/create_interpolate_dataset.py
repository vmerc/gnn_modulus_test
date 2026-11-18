#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import pickle
import sys
import os
import argparse
from tqdm import tqdm  # For progress bars

# Import custom modules
# Adjust the import paths based on your project structure
project_path = os.path.abspath(os.path.join(os.getcwd(), '..', ''))
sys.path.append(project_path)

try:
    from python.python_code.data_manip.extraction.telemac_file import TelemacFile
    from python.create_dgl_dataset import add_mesh_info
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    sys.exit(1)

def find_file_with_extension(directory, extension):
    """
    Finds the first file with the given extension in the specified directory.
    """
    for file in os.listdir(directory):
        if file.endswith(extension):
            return os.path.join(directory, file)
    return None

def load_pkl_file(filepath):
    """
    Loads a .pkl file using pickle.
    """
    with open(filepath, 'rb') as fp:
        data = pickle.load(fp)
    return data

def save_interpolated_data(data, filepath):
    """
    Saves interpolated data using pickle.
    """
    with open(filepath, 'wb') as fp:
        pickle.dump(data, fp)

def interpolate_data(data_list, X_coarse, triangulation, X_fine):
    """
    Interpolates data from a list of tuples (a_i, b_i) onto X_coarse using the provided triangulation.

    Parameters:
    - data_list: List of tuples (a_i, b_i)
    - X_coarse: Numpy array of shape (n_points, 2) representing coarse mesh nodes.
    - triangulation: Delaunay triangulation object based on fine mesh.
    - X_fine: Numpy array of fine mesh nodes (for validation)

    Returns:
    - List of tuples containing interpolated (a_i_interpolated, b_i_interpolated).
    """
    interpolated_results = []
    for idx, (a_i, b_i) in enumerate(data_list):
        print(f"Interpolating data {idx+1}/{len(data_list)}")
        # Ensure a_i and b_i are C-contiguous
        a_i = np.ascontiguousarray(a_i)
        b_i = np.ascontiguousarray(b_i)

        # Validate shapes
        assert a_i.shape[0] == X_fine.shape[0], f"Mismatch between a_i and X_fine in data {idx+1}"
        assert b_i.shape[0] == X_fine.shape[0], f"Mismatch between b_i and X_fine in data {idx+1}"

        # Interpolate a_i (all columns at once)
        interpolator_a = LinearNDInterpolator(triangulation, a_i)
        a_i_interpolated = interpolator_a(X_coarse)

        # Interpolate b_i (all columns at once)
        interpolator_b = LinearNDInterpolator(triangulation, b_i)
        b_i_interpolated = interpolator_b(X_coarse)

        # Handle potential NaNs resulting from extrapolation
        a_i_interpolated = np.nan_to_num(a_i_interpolated, nan=0.0)
        b_i_interpolated = np.nan_to_num(b_i_interpolated, nan=0.0)

        # Store interpolated results for this tuple
        interpolated_results.append((a_i_interpolated, b_i_interpolated))

    return interpolated_results

def main(fine_mesh_dir, coarse_mesh_dir, pkl_files_list, output_format='pkl'):
    """
    Main function to perform interpolation of specified .pkl files from fine mesh to coarse mesh.

    Parameters:
    - fine_mesh_dir: Path to the fine mesh directory containing .slf and .pkl files.
    - coarse_mesh_dir: Path to the coarse mesh directory containing .slf file.
    - pkl_files_list: List of .pkl filenames to process.
    - output_format: Format to save interpolated data ('pkl' by default).
    """
    # Validate input directories
    if not os.path.isdir(fine_mesh_dir):
        print(f"Fine mesh directory does not exist: {fine_mesh_dir}")
        sys.exit(1)
    if not os.path.isdir(coarse_mesh_dir):
        print(f"Coarse mesh directory does not exist: {coarse_mesh_dir}")
        sys.exit(1)

    # Locate .slf files
    fine_slf = find_file_with_extension(fine_mesh_dir, '.slf')
    if not fine_slf:
        print(f"No .slf file found in fine mesh directory: {fine_mesh_dir}")
        sys.exit(1)

    coarse_slf = find_file_with_extension(coarse_mesh_dir, '.slf')
    if not coarse_slf:
        print(f"No .slf file found in coarse mesh directory: {coarse_mesh_dir}")
        sys.exit(1)

    print(f"Fine mesh file: {fine_slf}")
    print(f"Coarse mesh file: {coarse_slf}")

    # Load meshes
    res_fine_mesh = TelemacFile(fine_slf)
    res_coarse_mesh = TelemacFile(coarse_slf)

    X_fine, triangles_fine = add_mesh_info(res_fine_mesh)
    X_coarse, triangles_coarse = add_mesh_info(res_coarse_mesh)

    # Precompute the triangulation
    print("Computing Delaunay triangulation for fine mesh...")
    triangulation = Delaunay(X_fine)
    print("Triangulation complete.")

    # Validate and collect the specified .pkl files
    all_pkl_files = os.listdir(fine_mesh_dir)
    available_pkl_files = [f for f in all_pkl_files if f.endswith('.pkl')]

    # If pkl_files_list is empty, process all available .pkl files
    if not pkl_files_list:
        pkl_files_to_process = available_pkl_files
    else:
        # Check that each specified .pkl file exists in the fine mesh directory
        pkl_files_to_process = []
        for pkl_file in pkl_files_list:
            if pkl_file not in available_pkl_files:
                print(f"Warning: Specified .pkl file not found in fine mesh directory: {pkl_file}")
            else:
                pkl_files_to_process.append(pkl_file)
        if not pkl_files_to_process:
            print("No valid .pkl files to process. Exiting.")
            sys.exit(1)

    print(f"Processing {len(pkl_files_to_process)} .pkl files:")
    for pkl_file in pkl_files_to_process:
        print(f" - {pkl_file}")

    # Loop over each .pkl file
    for pkl_file in tqdm(pkl_files_to_process, desc="Processing .pkl files"):
        pkl_path = os.path.join(fine_mesh_dir, pkl_file)
        data_list = load_pkl_file(pkl_path)

        # Process data_list, which is a list of tuples (a_i, b_i)
        print(f"Interpolating data from {pkl_file}...")
        interpolated_data = interpolate_data(data_list, X_coarse, triangulation, X_fine)
        print(f"Interpolation complete for {pkl_file}.")

        # Construct output filename
        base_name = os.path.splitext(pkl_file)[0]
        output_filename = f"{base_name}_interpolated.pkl"
        output_path = os.path.join(coarse_mesh_dir, output_filename)

        # Save the interpolated data
        save_interpolated_data(interpolated_data, output_path)
        print(f"Saved interpolated data to: {output_path}")

    print("All specified interpolated datasets have been saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate specified .pkl files from fine mesh to coarse mesh.")
    parser.add_argument("fine_mesh_directory_path", type=str, help="Path to the fine mesh directory containing .slf and .pkl files.")
    parser.add_argument("coarse_mesh_directory_path", type=str, help="Path to the coarse mesh directory containing .slf file.")
    parser.add_argument("--pkl_files", nargs='+', default=None, help="List of .pkl filenames to process. If not provided, all .pkl files in the fine mesh directory will be processed.")
    parser.add_argument("--output_format", type=str, default="pkl", help="Format to save interpolated data (default: pkl).")

    args = parser.parse_args()

    main(args.fine_mesh_directory_path, args.coarse_mesh_directory_path, args.pkl_files, args.output_format)
