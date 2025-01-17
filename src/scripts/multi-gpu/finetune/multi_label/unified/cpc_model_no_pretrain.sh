#!/bin/bash
#SBATCH --account=def-x
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1        # Request GPU "generic resources"
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4   # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=48000M             # This requests the nodes entire memory
#SBATCH --time=24:00:00     # DD-HH:MM:SS
#SBATCH --output=cpc_model_no_pretrain-%N.out

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
cp -R /path/to/Unified_Test_Runs/* "$SLURM_TMPDIR"

for i in {1..10}
do
  # Torch run will naturally access the last checkpoint if interrupted
  python -m src.patchECG_finetune \
            --lr=0.001 \
            --save_every=20 \
            --root_path="$SLURM_TMPDIR"/Unified_Dataset_"$i"/Unified_Dataset \
            --num_workers=4                             \
            --batch_size=256 \
            --metric_by_class \
            --data_augmentation=test_time_aug_cpc \
            --trafos=None \
            --class_token=cls_token  \
            --n_epochs_finetune=40 \
            --n_epochs_finetune_head=0 \
            --min_class_size=15 \
            --shared_embedding \
            --diagnostic_class=all \
            --layer_decay=1. \
            --weight_decay=0.001 \
            --head_dropout=0. \
            --dropout=0. \
            --dset_finetune=unified \
            --no-focal_loss \
            --model=cpc \
            --no-scheduler \
            --custom_lead_selection=all_leads \
            --model_selection_metric=valid_AUROC \
            --no-bootstrapping \
            --iterations=5000 \
            --model_name=cpc_model_no_pretrain_"$i"
done
