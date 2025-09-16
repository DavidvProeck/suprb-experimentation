#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --nodelist=oc-compute03

JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation

export PYTHONPATH=${PYTHONPATH}:${JOB_DIR}
export PYTHONPATH=$JOB_DIR:$JOB_DIR/src/suprb:$PYTHONPATH

echo "========== DEBUG INFO =========="
echo "Job ID: $SLURM_JOB_ID"
echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "================================"

experiment="runs/comparisons/nsga2/suprb_tuning_nsga2_volume.py"
dataset="airfoil_self_noise"
study_name="SupRB-Airfoil-NSGA2-Volume"

srun nix-shell $JOB_DIR/slurm/default_312.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p $dataset -j $SLURM_JOB_ID"
