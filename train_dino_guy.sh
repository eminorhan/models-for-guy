#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=dino_guy_i
#SBATCH --output=dino_guy_i_%A_%a.out

module purge
module load cuda-11.4

# S
# python -u -m torch.distributed.launch --nproc_per_node=4 /misc/vlgscratch4/LakeGroup/emin/ssl/train_dino_epoch.py --epochs 300 --use_fp16 True --arch resnext50_32x4d --batch_size_per_gpu 140 --optimizer adamw --weight_decay 0.000 --weight_decay_end 0.000 --clip_grad 1.0 --global_crops_scale 0.15 1 --local_crops_scale 0.05 0.15 --lr 0.0003 --min_lr 0.0003 --print_freq 100 --cache_path '/misc/vlgscratch4/LakeGroup/emin/ssl/cache/S_1fps.pth' --seed 1 --fraction 1.0 --freeze_last_layer 0 --output_dir /misc/vlgscratch4/LakeGroup/emin/ssl/guy_model_s --data_dirs '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_1fps'

# ImageNet
python -u -m torch.distributed.launch --nproc_per_node=4 /misc/vlgscratch4/LakeGroup/emin/ssl/train_dino_epoch.py --epochs 300 --use_fp16 True --arch resnext50_32x4d --batch_size_per_gpu 140 --optimizer adamw --weight_decay 0.000 --weight_decay_end 0.000 --clip_grad 1.0 --global_crops_scale 0.15 1 --local_crops_scale 0.05 0.15 --lr 0.0003 --min_lr 0.0003 --print_freq 100 --cache_path '/misc/vlgscratch4/LakeGroup/emin/ssl/cache/imagenet_train.pth' --seed 1 --fraction 1.0 --freeze_last_layer 1 --output_dir /misc/vlgscratch4/LakeGroup/emin/ssl/guy_model_i --data_dirs '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/train'

echo "Done"
