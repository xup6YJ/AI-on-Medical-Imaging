#!/bin/bash


python infer.py \
--patch_x 128 \
--patch_y 128 \
--patch_z 8 \
--patch_overlap_x 0 \
--patch_overlap_y 0 \
--patch_overlap_z 2 \
--dataset COVID \
--model DS_COVID_seg \
--deep_supervision \
--cv 1 \
--cpt_path /media/yclin/3TBNAS/Medical-AI/LAB3/DS_COVID_seg_COVID.cpts/DS_COVID_seg20230507_2129_DiceBCELoss_lr0.001_kaiming-init_200epochs_bs5_p128-128-8_o0-0-2_COVID_reduce-lr-on-plateau_curr_cv1/cv1_epoch136_batch5984_e136.cpt \
--output_dir /media/yclin/3TBNAS/Medical-AI/LAB3/DS_COVID_seg_COVID.cpts/DS_COVID_seg20230507_2129_DiceBCELoss_lr0.001_kaiming-init_200epochs_bs5_p128-128-8_o0-0-2_COVID_reduce-lr-on-plateau_curr_cv1


