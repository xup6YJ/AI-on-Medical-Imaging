# -*- coding: utf-8 -*-
import os
import json
import math
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_1samp
import warnings
from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tF
from torch.utils.tensorboard import SummaryWriter
import torchio as tio

from loss import *
from utils import *
from model_2 import *
from performance import measurement
from weights_initalization import *
from dset_io import CTDataset, SampleProbabilityMap, CTTestDataset

if torch.cuda.is_available:
    device = 'cuda'
else:
    device = 'cpu'

# random affine for test time augmentation
test_time_transform = tio.transforms.Compose([
    tio.transforms.RandomFlip(axes='LR', flip_probability=0.5),
    tio.transforms.RandomAffine(
        scales=(0, 0, 0),
        degrees=(0, 0, 10),
        translation=(10, 10, 0))
])


# def interhemispheric(model):
#     if hasattr(model, 'compare_left_right'):
#         return model.compare_left_right
#     else:
#         return model.module.compare_left_right


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
            images = patches_batch['image'][tio.DATA].to(device)

            # if interhemispheric(model):
            #     images_flip = patches_batch['image_flip'][tio.DATA].to(device)
            #     _, outputs = model(images, images_flip)
            # else:
            #     _, outputs = model(images)

            if not args.deep_supervision:
                outputs = model(images)
            else:
                output1, output2, outputs = model(images)

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
    if len(args.task_name):
        return args.task_name

    if args.continue_training:
        task_name = f'cont_epoch{args.continue_epoch}_{args.model}'
    else:
        task_name = args.model

    if args.cgm:
        task_name += f'_CGM'
        if args.cgm_weight < 1:
            task_name += f'-{args.cgm_weight}'

    # task_name += f'_in{args.in_channel}_out{args.out_channel}_init{args.init_ch}'

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


def create_model(args):
    model = eval(generate_model_string(args))
    return model


def train3d(args):
    patch_size = (args.patch_x, args.patch_y, args.patch_z)
    patch_overlap = (args.patch_overlap_x, args.patch_overlap_y, args.patch_overlap_z)
    task_name = get_task_name(args)

    cpt_dir = os.path.join(args.cpt_dir, f'{task_name}_cv{args.cv}')
    os.makedirs(cpt_dir, exist_ok=True)
    logger = SummaryWriter(os.path.join(args.log_dir, f'{task_name}_cv{args.cv}'))

    # model creation
    # model = create_model(args)
    # model = nn.DataParallel(model, device_ids=[0])

    model = eval(f'{args.model}()')
    model = model.to(device)

    # loss function
    if args.criterion == 'UnifiedFocalLoss':
        criterion = UnifiedFocalLoss(weight=args.uf_weight, delta=args.uf_delta, gamma=args.uf_gamma)
    elif args.criterion == 'FocalTverskyLoss':
        criterion = FocalTverskyLoss(alpha=args.ft_alpha, beta=args.ft_beta, gamma=args.ft_gamma)
    else:
        criterion = eval(f'{args.criterion}()')

    if args.bd_loss:
        boundary_loss = SurfaceLoss()

    if args.cgm:
        cgm_criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)
    elif args.exponential_lr:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    elif args.step_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.9)

    # create dataset and dataloader
    root_dir = f'../{args.dataset}/cv{args.cv}'
    train_dataset = CTDataset(mode='train',
                        no_noisy=args.no_noisy,
                        # flip_image=model.compare_left_right,
                        curriculum=args.curriculum)
    # val_dataset = CTDataset(mode='val', flip_image=model.compare_left_right)
    val_dataset = CTDataset(mode='val')    
    # weighted sampler (curriculum learning) or uniform sampler
   
    if args.curriculum:
        train_sampler = tio.data.WeightedSampler(patch_size, 'prob_map')
    else:
        train_sampler = tio.data.UniformSampler(patch_size)
    train_patches_queue = tio.Queue(
        train_dataset,
        args.queue_length,
        args.samples_per_volume,
        sampler=train_sampler,
        num_workers=0)
    
    train_loader = DataLoader(dataset=train_patches_queue, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f'[cv{args.cv}] There are {len(train_dataset):4d} subjects in train set.')
    print(f'[cv{args.cv}] There are {len(val_dataset):4d} subjects in val set.')

    batch_done = 0
    if args.continue_training:
        batch_done = args.continue_batch_done
    epoch_pbar = tqdm(range(1, args.num_epochs+1))
    model.zero_grad()
    best_valid_loss = float('inf')

    #Training
    for epoch in epoch_pbar:
        print('\n-'*10, ' Epoch ', epoch, '-'*10)
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
        epoch_measure = {
            'loss': 0,
            'seg_loss': 0
        }
        if args.bd_loss:
            epoch_measure['bd_loss'] = 0
        if args.cgm:
            epoch_measure['cgm_loss'] = 0

        batch_pbar = tqdm(train_loader)
        for batch_idx, data in enumerate(batch_pbar):
            # images = data['image'][tio.DATA].to(device)
            images = data['image'][tio.DATA].float().to(device)
            masks = data['label'][tio.DATA].to(device)
            masks = masks * data['icv'][tio.DATA].to(device)
            if args.cgm:
                has_roi = (masks.sum(dim=(1, 2, 3, 4))>0).long()

            # model prediction
            # if interhemispheric(model):
            #     images_flip = data['image_flip'][tio.DATA].to(device)
            #     pred_has_roi, outputs = model(images, images_flip)
            # else:
            #     pred_has_roi, outputs = model(images)

            if not args.deep_supervision:
                outputs = model(images)
            else:
                output1, output2, outputs = model(images)


            # segmentation loss, boundary loss, multi-task loss
            if not args.deep_supervision:
                seg_loss = criterion(outputs, masks)
            else:
                loss1 = criterion(output1, masks)
                loss2 = criterion(output2, masks)
                loss3 = criterion(outputs, masks)
                seg_loss = 0.25*loss1 + 0.25*loss2 + 0.5*loss3

            loss = seg_loss
            if args.bd_loss:
                bd_loss = boundary_loss(outputs, data['dist_map'][tio.DATA].to(device))
                loss = loss + args.bd_loss_weight*bd_loss
            if args.cgm:
                cgm_loss = cgm_criterion(pred_has_roi, has_roi)
                loss = loss + args.cgm_weight*cgm_loss

            batch_done += 1
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # write to tensorboard logger (batch)
            measure = measurement(outputs, masks)
            for k in measure:
                if epoch_measure.get(k) is None:
                    epoch_measure[k] = 0
                epoch_measure[k] += measure[k]

            logger.add_scalar('batch/loss', loss.item(), batch_done)
            logger.add_scalar('batch/seg_loss', seg_loss.item(), batch_done)
            epoch_measure['loss'] += loss.item()
            epoch_measure['seg_loss'] += seg_loss.item()
            if args.bd_loss:
                logger.add_scalar('batch/bd_loss', bd_loss.item(), batch_done)
                epoch_measure['bd_loss'] += bd_loss.item()
            if args.cgm:
                logger.add_scalar('batch/cgm_loss', cgm_loss.item(), batch_done)
                epoch_measure['cgm_loss'] += cgm_loss.item()

            batch_pbar.set_description(f'[train] [e:{epoch}/{args.num_epochs}] [b:{batch_idx+1}/{len(train_loader)}] loss: {loss.item():.4f}')

        # save model weight (epoch)
        # torch.save({
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict()
        #     },
        #     os.path.join(cpt_dir, f'cv{args.cv}_epoch{epoch}_batch{batch_done}_e{epoch}.cpt'))

        logger.add_scalar('epoch/lr', get_lr(optimizer), epoch)

        # write to tensorboard logger (epoch)
        for k in epoch_measure:
            epoch_measure[k] /= len(train_loader)
        for scalar in ['acc', 'iou', 'tpr', 'tnr', 'dsc', 'ppv', 'loss', 'seg_loss']:
            logger.add_scalar(f'train/{scalar}', epoch_measure[scalar], epoch)
        if args.bd_loss:
            logger.add_scalar('train/bd_loss', epoch_measure['bd_loss'], epoch)
        if args.cgm:
            logger.add_scalar('train/cgm_loss', epoch_measure['cgm_loss'], epoch)

        measure_val = evaluate3d(model, 
                                 criterion, val_dataset, (256, 256, 8), (0, 0, 2), 'val', 
                                 args.batch_size, args.test_time_aug, args.bd_loss_weight)
        
        for scalar in ['acc', 'iou', 'tpr', 'tnr', 'dsc', 'ppv', 'loss', 'seg_loss', 'bd_loss']:
            logger.add_scalar(f'val/{scalar}', measure_val[scalar], epoch)

        if args.reduce_lr_on_plateau:
            scheduler.step(epoch_measure['dsc'])
        elif args.exponential_lr or args.step_lr:
            scheduler.step()

        # save best model weight (epoch/ val loss)
        current_valid_loss = measure_val['loss']
        print(f"\ncurrent_valid_loss: {current_valid_loss:.4f}")
        print(f"Best validation loss: {best_valid_loss:.4f}")
        if current_valid_loss < best_valid_loss:
            best_valid_loss = current_valid_loss
            print(f"current_valid_loss < Best validation loss")
            print(f"Saving best model for epoch: {epoch}\n")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': criterion,
                }, 
                os.path.join(cpt_dir, f'cv{args.cv}_epoch{epoch}_batch{batch_done}_e{epoch}.cpt'))
        print("Save path: ", os.path.join(cpt_dir, f'cv{args.cv}_epoch{epoch}_batch{batch_done}_e{epoch}.cpt')) 

        epoch_pbar.set_description(f'[train] [e:{epoch}/{args.num_epochs}] avg.loss: {epoch_measure["loss"]:.4f} avg.sensitivity: {epoch_measure["tpr"]:.4f} avg.acc: {epoch_measure["acc"]:.4f} avg.dsc: {epoch_measure["dsc"]:.4f}')


def evaluate3d(model, criterion, dataset, patch_size, patch_overlap, tqdm_desc, batch_size=8, tta=False, bd_loss_weight=0.2):
    model.eval()
    totol_measure = {
        'seg_loss': 0,
        'bd_loss': 0,
        'loss': 0
    }
    subj_pbar = tqdm(dataset)
    for subj_idx, subj in enumerate(subj_pbar):
        if tta:
            outputs, var, mean, t, p = test_time_aug(model, subj, patch_size, patch_overlap)
        else:
            outputs = test(model, subj, patch_size, patch_overlap)
        masks = subj['label'][tio.DATA]
        outputs = outputs.unsqueeze(dim=0)
        masks = masks.unsqueeze(dim=0)
        seg_loss = criterion(outputs, masks)
        # boundary_loss = SurfaceLoss()
        # bd_loss = boundary_loss(outputs, subj['dist_map'][tio.DATA].unsqueeze(dim=0))
        measure = measurement(outputs, masks)
        for k in measure:
            if totol_measure.get(k) is None:
                totol_measure[k] = 0
            totol_measure[k] += measure[k]
        totol_measure['seg_loss'] += seg_loss.cpu().item()
        # totol_measure['bd_loss'] += bd_loss.cpu().item()
        # totol_measure['loss'] += (seg_loss.cpu().item() + bd_loss_weight*bd_loss.cpu().item())
        totol_measure['loss'] += seg_loss.cpu().item()

        subj_pbar.set_description(f'[eval-{tqdm_desc}] [b:{subj_idx+1}/{len(dataset)}] loss: {seg_loss.item():.4f}')
    for k in totol_measure:
        totol_measure[k] /= len(dataset)
    return totol_measure


def test3d(args):
    output_dir = args.output_dir
    if args.test_time_aug:
        output_dir += f'_tta-n{args.test_time_aug_n}'
    os.makedirs(output_dir, exist_ok=True)

    patch_size = (args.patch_x, args.patch_y, args.patch_z)
    patch_overlap = (args.patch_overlap_x, args.patch_overlap_y, args.patch_overlap_z)

    model = create_model(args)
    # model = nn.DataParallel(model, device_ids=[0])
    model = model.to(device)
    model.load_state_dict(torch.load(args.cpt_path, map_location=device)['model'])
    model.eval()

    root_dir = f'../{args.dataset}/cv{args.cv}'
    # dataset = CTDataset(f'{root_dir}/test', mode='val', flip_image=interhemispheric(model))
    dataset = CTDataset(mode='val')
    # dataset = AISTestDataset('../AISD_test')
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
            measure = measurement(outputs.unsqueeze(dim=0), subj['label'][tio.DATA].unsqueeze(dim=0))
            total_measure[subj['name']] = measure
            for k in measure:
                if k in avg_measure:
                    avg_measure[k] += measure[k]
                else:
                    avg_measure[k] = measure[k]
            outputs = outputs[-1,...].float()
            for z in range(slice_num):
                if args.test_time_aug:
                    plot_slice(
                        image=subj['image'][tio.DATA][...,z],
                        mask=subj['label'][tio.DATA][...,z],
                        output=outputs[...,z],
                        prob=p[1,...,z],
                        save_dir=f'{output_dir}/{subj["name"]}',
                        save_fn=f'{subj["name"]}_{z}_seg.jpg')
                else:
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
                    plot_slice(
                        image=subj['image'][tio.DATA][...,z],
                        mask=None,
                        output=outputs[...,z],
                        prob=p[1,...,z],
                        save_dir=f'{output_dir}/{subj["name"]}',
                        save_fn=f'{subj["name"]}_{z}_seg.jpg')
                else:
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
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()
    # for model
    parser.add_argument('--model', type=str, default='COVID_seg')
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--out_channel', type=int, default=2)
    parser.add_argument('--init_ch', type=int, default=64)
    parser.add_argument('--deconv', action='store_true', default=False)
    parser.add_argument('--weight_init', type=str, default='kaiming')
    parser.add_argument('--cr_cpt_path', type=str)
    parser.add_argument('--deep_supervision', action='store_true', default=False)

    # for UNet3P
    parser.add_argument('--reduce_ch', type=int, default=64)

    # for RUNet, R2UNet
    parser.add_argument('--num_rcnn', type=int, default=2)
    parser.add_argument('--t', type=int, default=2)

    # for classification guided module (multi-task)
    parser.add_argument('--cgm', action='store_true', default=False)
    parser.add_argument('--cgm_weight', type=float, default=0.2)

    # for focal tversky loss
    parser.add_argument('--ft_alpha', type=float, default=0.7)
    parser.add_argument('--ft_beta', type=float, default=0.3)
    parser.add_argument('--ft_gamma', type=float, default=0.75)

    # for unified focal loss
    parser.add_argument('--uf_weight', type=float, default=0.5)
    parser.add_argument('--uf_delta', type=float, default=0.6)
    parser.add_argument('--uf_gamma', type=float, default=0.2)

    # for boundary loss
    parser.add_argument('--bd_loss', action='store_true', default=False)
    parser.add_argument('--bd_loss_weight', type=float, default=0.2)

    # for training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--criterion', type=str, default='DiceBCELoss')
    parser.add_argument('--lr', type=float, default=1e-3)

    # for lr scheduler
    parser.add_argument('--reduce_lr_on_plateau', action='store_true', default=False)
    parser.add_argument('--exponential_lr', action='store_true', default=False)
    parser.add_argument('--step_lr', action='store_true', default=False)

    # for dataloader
    parser.add_argument('--dataset', type=str, default='COVID')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--patch_x', type=int, default=128)
    parser.add_argument('--patch_y', type=int, default=128)
    parser.add_argument('--patch_z', type=int, default=8)
    parser.add_argument('--patch_overlap_x', type=int, default=0)
    parser.add_argument('--patch_overlap_y', type=int, default=0)
    parser.add_argument('--patch_overlap_z', type=int, default=2)
    parser.add_argument('--queue_length', type=int, default=1000)
    parser.add_argument('--samples_per_volume', type=int, default=16)

    parser.add_argument('--no_noisy', action='store_true', default=False)
    parser.add_argument('--curriculum', action='store_true', default=True)

    parser.add_argument('--continue_training', action='store_true', default=False)
    parser.add_argument('--continue_epoch', type=int)
    parser.add_argument('--continue_batch_done', type=int)
    parser.add_argument('--continue_cpt_path', type=str)

    parser.add_argument('--test_time_aug', action='store_true', default=False)

    parser.add_argument('--cpt_dir', type=str, default='COVID_cpts')
    parser.add_argument('--log_dir', type=str, default='COVID_logs')

    parser.add_argument('--task_name', type=str, default='COVID_seg')
    parser.add_argument('--task_name_suffix', type=str, default='')

    # args, unknown = parser.parse_known_args()
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))

    train3d(args)