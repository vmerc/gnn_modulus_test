#!/bin/sh

#SBATCH --job-name=aube-regular-proj
#SBATCH --output=aube-regular-proj-%j.out
#SBATCH --error=aube-regular-proj-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Si besoin, remplace cette ligne par un lancement via apptainer/singularity.
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/project_regular_dataset_from_slf.py"
