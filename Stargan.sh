#!/bin/bash
#SBATCH --job-name=cyto_sg2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=cyto.out
#SBATCH --error=cyto.err

set -euo pipefail

# User-configurable paths
WORKDIR=${WORKDIR:-$PWD}
SCRIPT=${SCRIPT:-Stargan3.py}
DATASET_DIR=${DATASET_DIR:-/path/to/cytology_dataset}
OUTROOT=${OUTROOT:-$WORKDIR/outputs/cyto_sg}

module load cuda/11.8.0 || true
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${OUTROOT}"
cd "${WORKDIR}"

EPOCHS=50
IMG=256
BS=8
LR=2e-4
CH=64
STYLE_DIM=64
Z_DIM=32
MAX_PER_FRAME=64
SEED=1337
W_ID=2.0
W_CYC=0.5
W_DIV=0.05
CLF_EPOCHS=3
W_PERC=0.2

srun -u python "${SCRIPT}" \
  --root "${DATASET_DIR}" \
  --labels "${DATASET_DIR}/labels.csv" \
  --outdir "${OUTROOT}" \
  --epochs "${EPOCHS}" \
  --img "${IMG}" \
  --bs "${BS}" \
  --lr "${LR}" \
  --ch "${CH}" \
  --sdim "${STYLE_DIM}" \
  --zdim "${Z_DIM}" \
  --maxpf "${MAX_PER_FRAME}" \
  --seed "${SEED}" \
  --device cuda \
  --w_id "${W_ID}" \
  --w_cyc "${W_CYC}" \
  --w_div "${W_DIV}" \
  --clf_epochs "${CLF_EPOCHS}" \
  --w_perc "${W_PERC}" \
  --multi_disc

echo "All done."
