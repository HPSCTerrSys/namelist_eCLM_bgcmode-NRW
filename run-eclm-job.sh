#!/usr/bin/env bash
#SBATCH --job-name=eCLM-NRW
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --account=training2538
#SBATCH --partition=batch
#SBATCH --time=1:00:00
#SBATCH --output=logs/%j.eclm.out
#SBATCH --error=logs/%j.eclm.err

ECLM_LOADENVS=/p/project1/training2538/$USER/TSMP2/bin/JUSUF_eCLM/jsc.2025.intel.psmpi
if [[ ! -f $ECLM_LOADENVS || -z "$ECLM_LOADENVS" ]]; then
  echo "ERROR: Loadenvs script '$ECLM_LOADENVS' does not exist."
  exit 1
fi

source $ECLM_LOADENVS

ECLM_EXE=/p/project1/training2538/$USER/TSMP2/bin/JUSUF_eCLM/bin/eclm.exe
if [[ ! -f $ECLM_EXE || -z "$ECLM_EXE" ]]; then
  echo "ERROR: eCLM executable '$ECLM_EXE' does not exist."
  exit 1
fi

# Set PIO log files
if [[ -z $SLURM_JOB_ID || "$SLURM_JOB_ID" == " " ]]; then
  LOGID=$(date +%Y-%m-%d_%H.%M.%S)
else 
  LOGID=$SLURM_JOB_ID
fi
mkdir -p logs timing/checkpoints
LOGDIR=$(realpath logs)
comps=(atm cpl esp glc ice lnd ocn rof wav)
for comp in ${comps[*]}; do
  LOGFILE="$LOGID.comp_${comp}.log"
  sed -i "s#diro.*#diro = \"$LOGDIR\"#" ${comp}_modelio.nml
  sed -i "s#logfile.*#logfile = \"$LOGFILE\"#" ${comp}_modelio.nml
done

# Run model
srun $ECLM_EXE
