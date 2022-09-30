import os
import json
import math
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_1samp
import time
from datetime import datetime

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

if torch.cuda.is_available:
    device = 'cuda'
else:
    device = 'cpu'

import yaml
import argparse



# random affine for test time augmentation
test_time_transform = tio.transforms.Compose([
    tio.transforms.RandomFlip(axes='LR', flip_probability=0.5),
    tio.transforms.RandomAffine(
        scales=(0, 0, 0),
        degrees=(0, 0, 10),
        translation=(10, 10, 0))
])

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def test(model, subj, patch_size, patch_overlap, batch_size=8):
    # patch-based inference
    grid_sampler = tio.inference.GridSampler(subj, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size, num_workers=4)
    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

    with torch.no_grad():
        for patches_batch in patch_loader:

            # images = patches_batch['image'][tio.DATA].to(device)
            images = patches_batch['image'][tio.DATA].type(torch.FloatTensor).to(device)
            outputs = model(images)
            
            locations = patches_batch[tio.LOCATION]
            aggregator.add_batch(outputs, locations)

    outputs = aggregator.get_output_tensor()
    return outputs

def test_time_aug(model, subj, patch_size, patch_overlap, batch_size=8, n=20):
    results = []
    for _ in range(n):
        aug_subj = test_time_transform(subj)
        outputs = test(model, aug_subj, patch_size, patch_overlap, batch_size)
        aug_subj.add_image(image=tio.ScalarImage(tensor=outputs), image_name='segmentation')
        aug_subj = aug_subj.apply_inverse_transform()
        results.append(aug_subj['segmentation'][tio.DATA].type(torch.float32))
    results = torch.stack(results)
    t, p = ttest_1samp(results.numpy(), 0.5, axis=0, alternative='greater')
    var, mean = torch.var_mean(results, dim=0, unbiased=True)
    outputs = results.sum(dim=0) / n
    return outputs, var, mean, t, p

# generate task name string
def get_task_name(args):
    if args.task_name is not None:
        return args.task_name

    if args.continue_training:
        task_name = f'cont_epoch{args.continue_epoch}_{args.model}'
    else:
        task_name = args.model

    #timestamp
    now = datetime.now() 
    dt_string = now.strftime("%Y%m%d_%H%M")
    task_name += dt_string

    if args.cgm:
        task_name += f'_CGM'
        if args.cgm_weight < 1:
            task_name += f'-{args.cgm_weight}'

    task_name += f'_in{args.in_channel}_out{args.out_channel}_init{args.init_ch}'

    if args.deconv:
        task_name += '_deconv'
    if args.model in ['R2UNet', 'RUNet']:
        task_name += f'_rcnn{args.num_rcnn}_t{args.t}'

    if args.criterion == 'UnifiedFocalLoss':
        task_name += f'_UFL-w{args.uf_weight}-d{args.uf_delta}-g{args.uf_gamma}'
    elif args.criterion == 'FocalTverskyLoss':
        task_name += f'_FTL-a{args.ft_alpha}-b{args.ft_beta}-g{args.ft_gamma}'
    else:
        task_name += f'_{args.criterion.split(".")[-1]}'

    if args.bd_loss:
        task_name += f'_bdloss-w{args.bd_loss_weight}'

    task_name += f'_lr{args.lr}_{args.weight_init}-init'
    task_name += f'_{args.num_epochs}epochs_bs{args.batch_size}'
    task_name += f'_p{args.patch_x}-{args.patch_y}-{args.patch_z}'
    task_name += f'_o{args.patch_overlap_x}-{args.patch_overlap_y}-{args.patch_overlap_z}'

    task_name += f'_{args.dataset}'
    if args.no_noisy:
        task_name += '_no-noisy'

    if args.reduce_lr_on_plateau:
        task_name += '_reduce-lr-on-plateau'
    elif args.exponential_lr:
        task_name += '_exp-lr'
    elif args.step_lr:
        task_name += '_step-lr'

    if args.curriculum:
        task_name += '_curr'

    task_name += args.task_name_suffix
    return task_name

def train3d(args):

    patch_size = (args.patch_x, args.patch_y, args.patch_z)
    patch_overlap = (args.patch_overlap_x, args.patch_overlap_y, args.patch_overlap_z)
    task_name = get_task_name(args)

    cpt_dir = os.path.join(args.cpt_dir, f'{task_name}_cv{args.cv}')
    os.makedirs(cpt_dir, exist_ok=True)
    logger = SummaryWriter(os.path.join(args.log_dir, f'{task_name}_cv{args.cv}'))

    model = DeepMedicNR_test()
    model = model.to(device)

    criterion = DiceLoss()

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=1e-4, momentum=0.6, centered=False, foreach=None)

    # weight initialization
    if args.continue_training:
        model.load_state_dict(torch.load(args.continue_cpt_path)['model'])
        optimizer.load_state_dict(torch.load(args.continue_cpt_path)['optimizer'])
    else:
        if args.weight_init == 'xavier':
            model.apply(weights_initalization_xavier)
        elif args.weight_init == 'kaiming':
            model.apply(weights_initalization_kaiming)
        elif args.weight_init == 'default':
            model.apply(weights_initalization_default)
        elif args.weight_init == 'cr':
            weight = torch.load(args.cr_cpt_path)['model']
            weight.pop('conv_1x1.weight')
            weight.pop('conv_1x1.bias')
            model.load_state_dict(weight, strict=False)
        elif args.weight_init is None:
            pass
        else:
            raise ValueError('Unknown weight initialization method')

    # learning rate scheduler (if any)
    if args.reduce_lr_on_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    elif args.exponential_lr:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    elif args.step_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.9)

    # create dataset and dataloader
    ## modify
    train_dataset = CTDataset(curriculum=args.curriculum, validation = False)
    val_dataset = CTDataset(validation = True)

    # weighted sampler (curriculum learning) or uniform sampler
    if args.curriculum:
        train_sampler = tio.data.WeightedSampler(patch_size, 'prob_map')
        print('Training in WeightedSampler')
    else:
        train_sampler = tio.data.UniformSampler(patch_size)
        print('Training in UniformSampler')

    train_patches_queue = tio.Queue(
        train_dataset,
        args.queue_length,  
        args.samples_per_volume,  #8
        sampler=train_sampler,
        num_workers=0)
    
    train_loader = DataLoader(dataset=train_patches_queue, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f'[cv{args.cv}] There are {len(train_dataset):4d} subjects in train set.')
    print(f'[cv{args.cv}] There are {len(val_dataset):4d} subjects in val set.')

    batch_done = 0
    if args.continue_training:
        batch_done = args.continue_batch_done

    # print('='*25, 'Start Training', '='*25)
    epoch_pbar = tqdm(range(1, args.num_epochs+1))
    model.zero_grad()
    best_valid_loss = float('inf')
    for epoch in epoch_pbar:

        print(f"-"*10, 'Epoch: ',epoch,  '-'*10, '\n')
        if args.continue_training and epoch<args.continue_epoch:
            continue

        # update ratio for weighted sampler (curriculum learning)
        if args.curriculum:
            cur_transform = train_dataset._transform
            for i in range(len(cur_transform)):
                if isinstance(cur_transform[i], SampleProbabilityMap):
                    t = math.exp(-8*(1-epoch/args.num_epochs)**2)
                    logger.add_scalar('train/t', t, epoch)
                    cur_transform.transforms[i] = SampleProbabilityMap(icv_weight=t, include=['prob_map'])
                    break

        model.train()

        # epoch_measure = {'loss': 0, 'seg_loss': 0}
        epoch_measure = {'loss': 0}
        # epoch_measure['loss'] = 0
        # epoch_measure['dice_loss'] = 0

        batch_pbar = tqdm(train_loader)
        for batch_idx, data in enumerate(batch_pbar):
            # print(f"-"*10, 'batch_idx',batch_idx,  '-'*10, '\n')
            # print(f"-"*10, 'One Batch Start Training', '-'*10, '\n')
            images = data['image'][tio.DATA].type(torch.FloatTensor).to(device)
            masks = data['label'][tio.DATA].type(torch.FloatTensor)
            masks = (masks * data['icv'][tio.DATA]).type(torch.FloatTensor).to(device)
            # print(f"-"*5, 'Getting Images and Masks', '-'*5)

            # if args.cgm:
            #     has_roi = (masks.sum(dim=(1, 2, 3, 4))>0).long()  ## ??

            # model prediction
            outputs = model(images)
            # print(f"-"*5, 'Model Training Prediction', '-'*5)

            # segmentation loss, boundary loss, multi-task loss
            seg_loss = criterion(outputs, masks)
            loss = seg_loss
            # print(f"\n loss: {loss:.4f}")

            batch_done += 1
            loss.backward()
            # print(f"-"*5, 'Backward', '-'*5)
            optimizer.step()
            # print(f"-"*5, 'Optimizer Step', '-'*5)
            model.zero_grad()
            # print(f"-"*5, 'Model Zerograd', '-'*5)

            # write to tensorboard logger (batch)
            measure = measurement(outputs, masks)
            # print(f"-"*5, 'Measurement', '-'*5)

            for k in measure:
                if epoch_measure.get(k) is None:
                    epoch_measure[k] = 0
                epoch_measure[k] += measure[k]

            logger.add_scalar('batch/loss', loss.item(), batch_done)
            # logger.add_scalar('batch/seg_loss', seg_loss.item(), batch_done)
            epoch_measure['loss'] += loss.item()
            # epoch_measure['seg_loss'] += seg_loss.item()

            #batch bar
            batch_pbar.set_description(f'[train] [e:{epoch}/{args.num_epochs}] [b:{batch_idx+1}/{len(train_loader)}] loss: {loss.item():.4f}')
            # print(f"-"*10, 'One Batch Finished Training\n', '-'*10)
        # torch.save({
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict()
        #     },
        #     os.path.join(cpt_dir, f'cv{args.cv}_epoch{epoch}_batch{batch_done}_e{epoch}.cpt'))

        logger.add_scalar('epoch/lr', get_lr(optimizer), epoch)

        # write to tensorboard logger (epoch)
        for k in epoch_measure:
            epoch_measure[k] /= len(train_loader)
        # for scalar in ['acc', 'iou', 'tpr', 'tnr', 'dsc', 'ppv', 'loss', 'seg_loss']:
        for scalar in ['acc', 'iou', 'tpr', 'tnr', 'dsc', 'ppv', 'loss']:    
            logger.add_scalar(f'train/{scalar}', epoch_measure[scalar], epoch)

        # Validation
        measure_val = evaluate3d(
            model, criterion, val_dataset, (128, 128, 8), (0, 0, 2), 'val',   #256*256*8
            args.batch_size, args.test_time_aug, args.bd_loss_weight)

        # save best model weight (epoch/ val loss)
        
        current_valid_loss = measure_val['loss']
        print(f"\ncurrent_valid_loss: {current_valid_loss:.4f}")
        print(f"\nBest validation loss: {best_valid_loss:.4f}")
        if current_valid_loss < best_valid_loss:
            best_valid_loss = current_valid_loss
            print(f"\ncurrent_valid_loss < Best validation loss")
            print(f"\nSaving best model for epoch: {epoch}\n")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': criterion,
                }, 
                os.path.join(cpt_dir, f'cv{args.cv}_epoch{epoch}_batch{batch_done}_e{epoch}.cpt'))

        # for scalar in ['acc', 'iou', 'tpr', 'tnr', 'dsc', 'ppv', 'loss', 'seg_loss', 'bd_loss']:
        for scalar in ['acc', 'iou', 'tpr', 'tnr', 'dsc', 'ppv', 'loss']:    
            logger.add_scalar(f'val/{scalar}', measure_val[scalar], epoch)

        if args.reduce_lr_on_plateau:
            scheduler.step(epoch_measure['dsc'])
        elif args.exponential_lr or args.step_lr:
            scheduler.step()

        epoch_pbar.set_description(f'[train] [e:{epoch}/{args.num_epochs}] avg. loss: {epoch_measure["loss"]:.4f}')


def evaluate3d(model, criterion, dataset, patch_size, patch_overlap, tqdm_desc, batch_size=8, tta=False, bd_loss_weight=0.2):
    model.eval()
    totol_measure = {
        # 'seg_loss': 0,
        # 'bd_loss': 0,
        'loss': 0
    }
    subj_pbar = tqdm(dataset)
    for subj_idx, subj in enumerate(subj_pbar):
        if tta:
            outputs, var, mean, t, p = test_time_aug(model, subj, patch_size, patch_overlap)
        else:
            outputs = test(model, subj, patch_size, patch_overlap)

        ########## training
        # images = data['image'][tio.DATA].type(torch.FloatTensor).to(device)
        # masks = data['label'][tio.DATA].type(torch.FloatTensor)
        # masks = (masks * data['icv'][tio.DATA]).type(torch.FloatTensor).to(device)
        ##########

        ########## original val
        # masks = subj['label'][tio.DATA]
        # outputs = outputs.unsqueeze(dim=0)
        # masks = masks.unsqueeze(dim=0)
        # seg_loss = criterion(outputs, masks)
        ########## 

        masks = subj['label'][tio.DATA].type(torch.FloatTensor)
        outputs = outputs.unsqueeze(dim=0)
        masks = masks.unsqueeze(dim=0)
        seg_loss = criterion(outputs, masks)

        measure = measurement(outputs, masks)
        for k in measure:
            if totol_measure.get(k) is None:
                totol_measure[k] = 0
            totol_measure[k] += measure[k]
        # totol_measure['seg_loss'] += seg_loss.cpu().item()
        totol_measure['loss'] += seg_loss.cpu().item()

        subj_pbar.set_description(f'[eval-{tqdm_desc}] [b:{subj_idx+1}/{len(dataset)}] loss: {seg_loss.item():.4f}')
    for k in totol_measure:
        totol_measure[k] /= len(dataset)
    return totol_measure


if __name__ == '__main__':

    #read yaml file
    yaml_path = 'ctl.yaml'
    with open(yaml_path,'r') as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
        args = argparse.Namespace(**yaml.load(file, Loader=yaml.FullLoader))

    # print(json.dumps(vars(args), indent=2))
    train3d(args)