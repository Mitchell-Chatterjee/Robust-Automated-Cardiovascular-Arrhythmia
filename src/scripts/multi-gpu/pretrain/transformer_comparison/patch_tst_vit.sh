#!/bin/bash
#SBATCH --account=def-x
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1        # Request GPU "generic resources"
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4   # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=48000M             # This requests the nodes entire memory
#SBATCH --time=24:00:00     # DD-HH:MM:SS
#SBATCH --output=patch_tst-%N.out

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

# Move data into temp memory for faster processing
cp -R /path/to/Unified_Dataset "$SLURM_TMPDIR"

# Torch run will naturally access the last checkpoint if interrupted
srun python -m src.patchtst_pretrain \
          --init_method tcp://$MAIN_NODE:3456   \
          --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))            \
          --lr=0.0005 \
          --save_every=20 \
          --dset_pretrain=unified \
          --root_path="$SLURM_TMPDIR"/Unified_Dataset \
          --num_workers=4 \
          --batch_size=256 \
          --n_epochs_pretrain=500 \
          --warmup_epochs=2 \
          --plot_every_n=1 \
          --distributed   \
          --shared_embedding \
          --d_model=128    \
          --n_heads=8       \
          --n_layers=6     \
          --d_ff=512          \
          --patch_len=20     \
          --stride=20        \
          --mask_ratio=0.4    \
          --dropout=0.1 \
          --head_dropout=0.1 \
          --weight_decay=0.05 \
          --data_augmentation=per_lead_aug \
          --custom_lead_selection=all_leads \
          --trafos=None \
          --model=patch_tst \
          --model_name=patch_tst
