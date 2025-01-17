#!/bin/bash
#SBATCH --account=def-x
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1        # Request GPU "generic resources"
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4   # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=48000M        # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=24:00:00      # DD-HH:MM:SS
#SBATCH --output=vanilla_vit_comparison.out

module load python/3.10 cuda cudnn

# Prepare virtualenv
source /path/to/bin/activate

# Use same number of threads as cpus
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


# Move to correct directory
cd /path/to/Masters-Research/ || exit

# Move data into temp memory for faster processing
cp -R /path/to/ptb-xl_1.0.3 "$SLURM_TMPDIR"


for i in {1..10}
do
  python    -m src.patchECG_finetune                    \
            --lr=0.001                                 \
            --save_every=20 \
            --root_path="$SLURM_TMPDIR"/ptb-xl_1.0.3 \
            --num_workers=4                             \
            --batch_size=128                             \
            --metric_by_class \
            --data_augmentation=none \
            --trafos=None \
            --class_token=cls_token                     \
            --shared_embedding \
            --min_class_size=15 \
            --diagnostic_class=all \
            --layer_decay=0.65 \
            --weight_decay=0.05 \
            --head_dropout=0. \
            --dropout=0. \
            --dset_finetune=ptb-xl \
            --no-focal_loss \
            --model=vanilla_vit \
            --d_model=128    \
            --pretrained_model_path=/path/to/pretrained_model.pt \
            --no-scheduler \
            --custom_lead_selection=all_leads \
            --n_epochs_finetune_head=20 \
            --n_epochs_finetune=70 \
            --n_heads=8       \
            --n_layers=6      \
            --d_ff=512          \
            --patch_len=50     \
            --stride=50        \
            --no-bootstrapping \
            --iterations=5000 \
            --model_name=vanilla_vit_comparison_"$i"
done