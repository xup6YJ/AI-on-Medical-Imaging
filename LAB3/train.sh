#!/bin/bash

python main.py \
--model COVID_seg \
--criterion DiceBCELoss \
--num_epochs 100 \
--batch_size 5 \
--queue_length 1000 \
--samples_per_volume 16 \
--cv 1 \
--dataset COVID \
--lr 1e-3 \
--patch_x 128 \
--patch_y 128 \
--patch_z 8 \
--patch_overlap_x 0 \
--patch_overlap_y 0 \
--patch_overlap_z 2 \
--cpt_dir COVID.cpts \
--log_dir COVID.logs
# --weight_init cr \
# --cr_cpt_path cr.cpts/ResUNet_in1_out1_init32_lr0.0002_kaiming-init_300epochs_bs8_p128-128-8_cr/epoch200_batch388000_e200.cpt
