#!/bin/bash
#SBATCH --account=def-x
#SBATCH --array=1-5%1       # 3 is the number of jobs in the chain. Gives approximately 12 epochs.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2        # Request GPU "generic resources"
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4   # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=97000M             # This requests the nodes entire memory
#SBATCH --time=24:00:00     # DD-HH:MM:SS
#SBATCH --output=patch_ecg-%N.out

module load python/3.10 cuda cudnn

# Prepare virtualenv
source /home/mitch/ENV2/bin/activate

# Use same number of threads as cpus
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export NCCL_BLOCKING_WAIT=1
export MAIN_NODE=$(hostname)
echo Hostname: $MAIN_NODE

# Move to correct directory
cd /path/to/Masters-Research/ || exit

# Move data into temp memory for faster processing
cp -R /path/to/CODE "$SLURM_TMPDIR"

# Torch run will naturally access the last checkpoint if interrupted
srun python -m src.patchECG_pretrain \
          --init_method tcp://$MAIN_NODE:3456   \
          --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))            \
          --lr=0.0015 \
          --save_every=20 \
          --dset_pretrain=CODE_Unannotated \
          --root_path="$SLURM_TMPDIR"/CODE \
          --num_workers=4 \
          --batch_size=100 \
          --n_epochs_pretrain=100 \
          --warmup_epochs=2 \
          --plot_every_n=1 \
          --distributed   \
          --shared_embedding \
          --d_model=768    \
          --n_heads=12       \
          --n_layers=12     \
          --d_ff=3072          \
          --patch_len=50     \
          --stride=50        \
          --mask_ratio=0.4    \
          --dropout=0. \
          --head_dropout=0. \
          --weight_decay=0.05 \
          --data_augmentation=none \
          --custom_lead_selection=all_leads \
          --trafos=None \
          --model=vanilla_vit \
          --model_name=patch_ecg
