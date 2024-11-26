#!/bin/bash
#SBATCH --account=def-x
#SBATCH --nodes=2
#SBATCH --gpus-per-node=v100l:2        # Request GPU "generic resources"
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=12   # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=64000M             # This requests the nodes entire memory
#SBATCH --time=00:30:00     # DD-HH:MM:SS

module load python/3.10 cuda cudnn

# Prepare virtualenv
source /path/to/bin/activate

# Use same number of threads as cpus
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export NCCL_BLOCKING_WAIT=1
export MAIN_NODE=$(hostname)
echo Hostname: $MAIN_NODE

# Move to correct directory
cd /path/to/Masters-Research/ || exit

# Torch run will naturally access the last checkpoint if interrupted
srun python -m src.patchECG_pretrain        \
      --init_method tcp://$MAIN_NODE:3456   \
      --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))            \
      --lr=0.0005                           \
      --save_every=1                        \
      --root_path=/path/to/CODE_test \
      --num_workers=8                       \
      --batch_size=64                       \
      --n_epochs_pretrain=15                \
      --plot_every_n=1                      \
      --distributed                         \
      #--model_name=