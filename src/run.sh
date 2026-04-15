#!/bin/bash
#SBATCH --job-name=cuda_check
#SBATCH --output=output/%j.out
#SBATCH --error=error/%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module load cuda-12.4
module load gcc-12.4.0

# print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"

# check GPU assigned
nvidia-smi

# test
./a.out

echo "Finished at: $(date)"
