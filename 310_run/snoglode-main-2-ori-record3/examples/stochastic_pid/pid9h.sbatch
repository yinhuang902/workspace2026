#!/bin/bash
#SBATCH --job-name=pid6h
#SBATCH --time=09:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=pid6h-%j.out
#SBATCH --error=pid6h-%j.err

set -euo pipefail

WORKDIR="/home/yinhuang/snoglode-main-2-ori-record3/examples/stochastic_pid"
PY="${WORKDIR}/stochastic_pid.py"

echo "JobID: $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "Work:  $WORKDIR"

cd "$WORKDIR"
ls -lah "$PY"

# If your code uses Gurobi, keep these
module purge
module load gurobi/13.0.0

# Threading (helps BLAS/OpenMP + often Gurobi if it respects threads)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Optional: if your code reads this env var, it can switch scenarios automatically
# export N_SCENARIOS=50

# micromamba env
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($HOME/bin/bin/micromamba shell hook -s bash)"
micromamba activate pidsp-sim-fir

echo "Python: $(which python)"
python -V

# Run script (limited by Slurm time)
python "$PY"

echo "End:   $(date)"
