# Mesh Transfer Regular

- `create_regular_mesh.ipynb` : cree un maillage regulier `.slf` a partir d'un maillage fin TELEMAC et projette `FOND` plus `FROTTEMENT`.
- `project_dataset_to_regular.ipynb` : projette un dataset existant `*_base.bin + *.pkl` sur le maillage regulier et ecrit un nouveau `*_base.bin` plus les fichiers `.pkl` projetes.
- `zero_shot_unroll_regular.ipynb` : teste un checkpoint existant en zero-shot sur le dataset regulier projete avec un unrolling simple sur une sequence.
