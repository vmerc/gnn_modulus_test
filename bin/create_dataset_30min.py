import os
import sys
import glob

project_path = os.path.abspath(os.path.join(os.getcwd(), '..', ''))
sys.path.append(project_path)
from python.create_dgl_dataset import create_dgl_dataset_chunked

data_folder = '/work/m24046/m24046mrcr/Test_Aube_regular/original_dataset'

mesh_file = os.path.join(data_folder, 'T2V6_KV4_BuseV2.geo')
cli_file = os.path.join(data_folder, 'cas.conlim')
dt_value = 1
ENFORCE_Q_BOUNDARY = False
ENFORCE_H_BOUNDARY = True

res_files = glob.glob(os.path.join(data_folder, '*.res'))

for res_file in res_files:
    try:
        base_name = os.path.splitext(os.path.basename(res_file))[0]
        res_list = [res_file]
        mesh_list = [mesh_file]
        cli_list = [cli_file]
        dt_list = [dt_value]
        dataset_name = base_name

        print(f"Création du dataset pour {res_file} avec le nom {dataset_name}")
        create_dgl_dataset_chunked(
            mesh_list,
            res_list,
            cli_list,
            dt_list,
            '/work/m24046/m24046mrcr/Test_Aube_regular/original_dataset',
            dataset_name,
            chunk_size=100,
            enforce_q_boundary=ENFORCE_Q_BOUNDARY,
            enforce_h_boundary=ENFORCE_H_BOUNDARY,
        )
    except Exception as e:
        print(f"Erreur lors de la création du dataset pour {res_file}: {e}")
