#!/bin/bash
#SBATCH --account=def-x
#SBATCH --array=1-7%1       # 3 is the number of jobs in the chain.
#SBATCH --nodes 1
#SBATCH --gres=gpu:1        # Request GPU "generic resources"
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12   # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=64000M        # This requests the nodes entire memory
#SBATCH --time=24:00:00     # DD-HH:MM:SS

module load python/3.10 cuda cudnn

# Prepare virtualenv
source /path/to/bin/activate

# Use same number of threads as cpus
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Move to correct directory
cd /path/to/Masters-Research/ || exit

# Execute code
python  -m src.scripts.unification.unify_datasets \
        --root_path_code=/path/to/CODE/               \
        --resample_freq=100                                 \
        --time_window=10                                    \
        --data_sub_path_code=training                       \
        --method=preprocess                                 \
        --code                                              \
