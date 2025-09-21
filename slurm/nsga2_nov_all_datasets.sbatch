#!/usr/bin/env bash
###############################################################################
# Slurm: array of per-dataset runs
###############################################################################
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --nodelist=oc-compute03
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --job-name=NSGA2Novelty            # shows up in squeue/sacct and filenames
#SBATCH --array=0-3                              # one task per dataset (see list below)
# Structured logs: output/<jobname>_<arrayid>_<taskid>.{txt,err}
#SBATCH --output=output/%x_%A_%a.txt
#SBATCH --error=output/%x_%A_%a.err
###############################################################################

set -euo pipefail

# ---------------------------------------------------------------------------
# 1) Per-job environment
# ---------------------------------------------------------------------------
JOB_DIR="/data/oc-compute03/$USER/suprb-experimentation"

export PYTHONPATH="${JOB_DIR}:${JOB_DIR}/src/suprb:${PYTHONPATH:-}"
experiment="runs/comparisons/nsga2/suprb_tuning_nsga2_novelty.py"

# ---------------------------------------------------------------------------
# 2) Map array-task ID → dataset name
# ---------------------------------------------------------------------------
datasets=(
  airfoil_self_noise
  concrete_strength
  combined_cycle_power_plant
  energy_cool
)

dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
study_name="NSGA2-${dataset}-Novelty-tuning"

# ---------------------------------------------------------------------------
# 3) Debug info
# ---------------------------------------------------------------------------
echo "========== DEBUG INFO =========="
echo "Date:        $(date)"
echo "Job Name:    ${SLURM_JOB_NAME:-}"
echo "Array IDs:   Job=${SLURM_ARRAY_JOB_ID:-}  Task=${SLURM_ARRAY_TASK_ID:-}"
echo "Node:        $(hostname)"
echo "PATH:        $PATH"
echo "PYTHONPATH:  $PYTHONPATH"
echo "Experiment:  $experiment"
echo "Dataset:     $dataset"
echo "Study:       $study_name"
echo "================================"

# ---------------------------------------------------------------------------
# 4) Launch the experiment
# ---------------------------------------------------------------------------
echo "[$(date)] Starting experiment on dataset '${dataset}' (array task ${SLURM_ARRAY_TASK_ID})"

srun nix-shell "${JOB_DIR}/slurm/default_312.nix" --command "
  PYTHONPATH='${JOB_DIR}:${JOB_DIR}/src/suprb:${PYTHONPATH}' \
  python '${JOB_DIR}/${experiment}' \
    -p '${dataset}' \
    -j '${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}' \
    -n '${study_name}'
"
