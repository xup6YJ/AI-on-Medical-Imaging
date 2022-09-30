import os
import json
import math
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_1samp

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tF
from torch.utils.tensorboard import SummaryWriter

import torchio as tio

from loss import *
from utils import *
from models import *
from performance import measurement
from weights_initalization import *
from dset_io import CTDataset, AISTestDataset
from dset_io import SampleProbabilityMap
from deepmedic_2 import *
from train import *

if torch.cuda.is_available:
    device = 'cuda'
else:
    device = 'cpu'

import yaml
import argparse


def test3d(args):
    output_dir = args.output_dir
    if args.test_time_aug:
        output_dir += f'_tta-n{args.test_time_aug_n}'
    os.makedirs(output_dir, exist_ok=True)

    patch_size = (args.patch_x, args.patch_y, args.patch_z)
    patch_overlap = (args.patch_overlap_x, args.patch_overlap_y, args.patch_overlap_z)

    # model = create_model(args)
    # model = nn.DataParallel(model, device_ids=[0])

    model = DeepMedicNR_test()  #modify
    model = model.to(device)
    model.load_state_dict(torch.load(args.cpt_path, map_location=device)['model'])
    model.eval()

    # root_dir = f'../{args.dataset}/cv{args.cv}'
    # dataset = CTDataset(f'{root_dir}/test', mode='val', flip_image=interhemispheric(model))
    dataset = AISTestDataset()
    print(f'There are {len(dataset):4d} subjects in test set.')

    subj_pbar = tqdm(dataset)
    total_measure = {}
    avg_measure = {}
    for subj_idx, subj in enumerate(subj_pbar):
        if args.test_time_aug:
            outputs, var, mean, t, p = test_time_aug(model, subj, patch_size, patch_overlap, n=args.test_time_aug_n)
            p = 1 - p
        else:
            outputs = test(model, subj, patch_size, patch_overlap)
            
        slice_num = subj['image'][tio.DATA].shape[-1]

        if subj.get('label'):
            # measure = measurement(outputs.unsqueeze(dim=0), subj['label'][tio.DATA].unsqueeze(dim=0))
            measure = measurement(outputs.unsqueeze(dim=0), subj['label'][tio.DATA].type(torch.FloatTensor).unsqueeze(dim=0))

            total_measure[subj['name']] = measure
            for k in measure:
                if k in avg_measure:
                    avg_measure[k] += measure[k]
                else:
                    avg_measure[k] = measure[k]
            outputs = outputs[-1,...].float()
            for z in range(slice_num):
                if args.test_time_aug:
                    if args.plot_image:
                        plot_slice(
                            image=subj['image'][tio.DATA][...,z],
                            mask=subj['label'][tio.DATA][...,z],
                            output=outputs[...,z],
                            prob=p[1,...,z],
                            save_dir=f'{output_dir}/{subj["name"]}',
                            save_fn=f'{subj["name"]}_{z}_seg.jpg')
                else:
                    if args.plot_image:
                        plot_slice(
                            image=subj['image'][tio.DATA][...,z],
                            mask=subj['label'][tio.DATA][...,z],
                            output=outputs[...,z],
                            prob=None,
                            save_dir=f'{output_dir}/{subj["name"]}',
                            save_fn=f'{subj["name"]}_{z}_seg.jpg')
        else:
            outputs = outputs[-1,...].float()
            for z in range(slice_num):
                if args.test_time_aug:
                    if args.plot_image:
                        plot_slice(
                            image=subj['image'][tio.DATA][...,z],
                            mask=None,
                            output=outputs[...,z],
                            prob=p[1,...,z],
                            save_dir=f'{output_dir}/{subj["name"]}',
                            save_fn=f'{subj["name"]}_{z}_seg.jpg')
                else:
                    if args.plot_image:
                        plot_slice(
                            image=subj['image'][tio.DATA][...,z],
                            mask=None,
                            output=outputs[...,z],
                            prob=None,
                            save_dir=f'{output_dir}/{subj["name"]}',
                            save_fn=f'{subj["name"]}_{z}_seg.jpg')

        subj_pbar.set_description(f'[test] subject:{subj_idx+1:>5}/{len(dataset)}')

    if len(total_measure):
        for k in avg_measure:
            avg_measure[k] /= len(total_measure)
        total_dice = [total_measure[subj]['dsc'] for subj in total_measure]
        avg_measure['dsc-var'] = np.var(total_dice)
        json.dump(total_measure, open(os.path.join(output_dir, 'total_measure.json'), 'w'), indent=2)
        json.dump(avg_measure, open(os.path.join(output_dir, 'avg_measure.json'), 'w'), indent=2)

if __name__ == '__main__':

    #read yaml file
    yaml_path = 'ctl_inf.yaml'
    with open(yaml_path,'r') as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
        args = argparse.Namespace(**yaml.load(file, Loader=yaml.FullLoader))

    # print(json.dumps(vars(args), indent=2))
    test3d(args)