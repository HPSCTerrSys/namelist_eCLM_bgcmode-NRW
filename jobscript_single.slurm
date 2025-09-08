#!/usr/bin/env bash
#SBATCH --job-name=nrw_300x300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --account=training2538
#SBATCH --partition=batch
#SBATCH --time=1:00:00
#SBATCH --output=logs/%j.eclm.nrw_300x300.out
#SBATCH --error=logs/%j.eclm.nrw_300x300.err

# Load environment
source loadenvs

# Run model
srun -n $SLURM_NTASKS eclm.exe
