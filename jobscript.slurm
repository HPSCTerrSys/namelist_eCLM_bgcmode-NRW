#!/usr/bin/env bash
#SBATCH --job-name=nrw_300x300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --account=training2538
#SBATCH --partition=batch
###SBATCH --reservation=training2433_day2
#SBATCH --time=2:30:00
#SBATCH --output=logs/%j.eclm_pdaf.nrw_300x300.out
#SBATCH --error=logs/%j.eclm_pdaf.nrw_300x300.err

# Load environment
source loadenvs

# Run model
srun -n $SLURM_NTASKS ./tsmp-pdaf -n_modeltasks 4 -screen 3 -filtertype 2 -subtype 1 -delt_obs 1 -rms_obs 0.02 -obs_filename ./obs/SM_CLM
