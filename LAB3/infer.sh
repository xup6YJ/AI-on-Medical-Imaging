#!/bin/bash

# 3-day_gt-1_5fold

python infer.py \
--in_channel 1 \ 
--out_channel 2 \ 
--init_ch 32 \ 
--patch_x 128 \ 
--patch_y 128 \ 
--patch_z 8 \ 
--patch_overlap_x 0 \ 
--patch_overlap_y 0 \ 
--patch_overlap_z 2 \
--dataset COVID \
--model UNet \
--cgm \
--cv 5 \
--cpt_path /media/yclin/3TBNAS/Medical-AI/LAB3/Unet_COVID.cpts/UNet20230505_1447_DiceBCELoss_lr0.001_kaiming-init_200epochs_bs5_p128-128-8_o0-0-2_COVID_reduce-lr-on-plateau_curr_cv1/cv1_epoch37_batch1628_e37.cpt \
--output_dir /media/yclin/3TBNAS/Medical-AI/LAB3/Unet_COVID.cpts/UNet20230505_1447_DiceBCELoss_lr0.001_kaiming-init_200epochs_bs5_p128-128-8_o0-0-2_COVID_reduce-lr-on-plateau_curr_cv1
