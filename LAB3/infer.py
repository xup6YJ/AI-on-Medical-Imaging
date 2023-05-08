# -*- coding: utf-8 -*-
import json
import warnings
from argparse import ArgumentParser

import torch
torch.cuda.set_device('cuda:0')

from train_io import test3d


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning) 

    parser = ArgumentParser()
    # for model
    parser.add_argument('--model', type=str, default='DS_COVID_seg')
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--out_channel', type=int, default=2)
    parser.add_argument('--init_ch', type=int, default=64)
    parser.add_argument('--deconv', action='store_true', default=True)

    # for UNet3P
    parser.add_argument('--reduce_ch', type=int, default=64)

    # for RUNet, R2UNet
    parser.add_argument('--num_rcnn', type=int, default=2)
    parser.add_argument('--t', type=int, default=2)

    # for deep supervision
    parser.add_argument('--deep_supervision', action='store_true', default=True)

    # for classification guided module (multi-task)
    parser.add_argument('--cgm', action='store_true', default=False)

    # for dataloader
    parser.add_argument('--dataset', type=str, default = 'COVID')
    parser.add_argument('--cv', type=int, default = 1)
    parser.add_argument('--patch_x', type=int, default = 128)
    parser.add_argument('--patch_y', type=int, default = 128)
    parser.add_argument('--patch_z', type=int, default = 8)
    parser.add_argument('--patch_overlap_x', type=int, default = 0)
    parser.add_argument('--patch_overlap_y', type=int, default = 0)
    parser.add_argument('--patch_overlap_z', type=int, default = 2)

    parser.add_argument('--test_time_aug', action='store_true', default=False)
    parser.add_argument('--test_time_aug_n', type=int, default=20)

    # parser.add_argument('--cpt_path', type=str, default='/media/yclin/3TBNAS/Medical-AI/LAB3/Unet_COVID.cpts/UNet20230505_1447_DiceBCELoss_lr0.001_kaiming-init_200epochs_bs5_p128-128-8_o0-0-2_COVID_reduce-lr-on-plateau_curr_cv1/cv1_epoch37_batch1628_e37.cpt')
    # parser.add_argument('--output_dir', type=str, default='/media/yclin/3TBNAS/Medical-AI/LAB3/Unet_COVID.cpts/UNet20230505_1447_DiceBCELoss_lr0.001_kaiming-init_200epochs_bs5_p128-128-8_o0-0-2_COVID_reduce-lr-on-plateau_curr_cv1')
    
    parser.add_argument('--cpt_path', type=str, default='/media/yclin/3TBNAS/Medical-AI/LAB3/DS_COVID_seg_COVID.cpts/DS_COVID_seg20230507_2129_DiceBCELoss_lr0.001_kaiming-init_200epochs_bs5_p128-128-8_o0-0-2_COVID_reduce-lr-on-plateau_curr_cv1/cv1_epoch136_batch5984_e136.cpt')
    parser.add_argument('--output_dir', type=str, default='/media/yclin/3TBNAS/Medical-AI/LAB3/DS_COVID_seg_COVID.cpts/DS_COVID_seg20230507_2129_DiceBCELoss_lr0.001_kaiming-init_200epochs_bs5_p128-128-8_o0-0-2_COVID_reduce-lr-on-plateau_curr_cv1')

    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))

    test3d(args)




