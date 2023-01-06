#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 19:30:10 2023

@author: yingchihlin
"""



import copy
import csv
import functools
import glob
import math
import random
import shutil
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2

import cc3d
import logging
import dicom2nifti
from PIL import Image
from tqdm import tqdm
import nibabel as nib
import dicom2nifti.settings as settings
from skimage.morphology import erosion, dilation, cube
from scipy.ndimage.morphology import binary_fill_holes


def dcm2nii(series_dir, output_path):
    dicom2nifti.dicom_series_to_nifti(series_dir, output_path, reorient_nifti=True)


def reorient(nii_path, output_path):
    img = nib.load(nii_path)
    data = img.get_fdata()
    header = img.header
    affine = img.affine

    z_height = data.shape[-1]
    mask = data > 600
    bone_slice = np.sum(mask, axis=(0, 1))
    z_top = np.argwhere(bone_slice>0)[-1,0] + 1
    top_to_center = 70 / header['pixdim'][3]
    if z_top-top_to_center > z_height/2:
        z_voxel = z_top - top_to_center
    else:
        z_voxel = z_height / 2
    x_voxel = 256
    y_voxel = 180

    ori_x = -1 * (x_voxel*affine[0,0] + y_voxel*affine[0,1] + z_voxel*affine[0,2])
    ori_y = -1 * (x_voxel*affine[1,0] + y_voxel*affine[1,1] + z_voxel*affine[1,2])
    ori_z = -1 * (x_voxel*affine[2,0] + y_voxel*affine[2,1] + z_voxel*affine[2,2])

    header['quatern_b'] = 0
    header['quatern_c'] = 0
    header['quatern_d'] = 0
    header['qoffset_x'] = 0
    header['qoffset_y'] = 0
    header['qoffset_z'] = 0
    header['srow_x'][-1] = ori_x
    header['srow_y'][-1] = ori_y
    header['srow_z'][-1] = ori_z
    header.set_slope_inter(1, 0)

    affine[0][-1] = ori_x
    affine[1][-1] = ori_y
    affine[2][-1] = ori_z

    nib.save(img, output_path)


def skull_strip(nii_path, output_ct_path, output_icv_path):
    intensity_thres = 100
    img = nib.load(nii_path)
    CT = img.get_fdata()
    CT_ori = np.array(CT)
    CT[CT<0] = 0
    CT[CT>=intensity_thres] = 0
    CT_ori[CT_ori<0] = 0
    CT_ori[CT_ori>intensity_thres] = intensity_thres
    CT_ori = CT_ori / intensity_thres
    mask = np.array(CT)
    mask[mask>0] = 1

    labels = cc3d.connected_components(mask, connectivity=6)
    stats = cc3d.statistics(labels)
    idx = np.argpartition(stats['voxel_counts'], -2)[-2:]
    idx = idx[np.argsort(stats['voxel_counts'][idx])]
    tissue_mask = labels==idx[0]
    tissue_mask = erosion(tissue_mask, footprint=cube(2))
    tissue_mask = dilation(tissue_mask, footprint=cube(2))

    rough_ICV_mask = np.zeros(CT.shape)
    for z in range(CT.shape[-1]):
        slice_tissue_mask = tissue_mask[...,z]
        if slice_tissue_mask.sum() == 0:
            continue
        cc = cc3d.connected_components(slice_tissue_mask, connectivity=8)
        stats = cc3d.statistics(cc)
        k = min(len(stats['voxel_counts']), 4)
        idx = np.argpartition(stats['voxel_counts'], -k)[-k:]
        idx = idx[np.argsort(stats['voxel_counts'][idx])]
        idx = idx[:3]
        slice_rough_ICV_mask = np.zeros(slice_tissue_mask.shape)
        for i in idx:
            if (cc==i).sum() < 1500:
                continue
            x, y = np.where((cc==i)>0)
            y_mid = (y.min()+y.max()) // 2
            if y_mid>340 or y_mid<150:
                continue
            x_mid = (x.min()+x.max()) // 2
            if x_mid<170 or x_mid>342:
                continue
            slice_rough_ICV_mask[cc==i] = slice_tissue_mask[cc==i]
        slice_rough_ICV_mask = binary_fill_holes(slice_rough_ICV_mask)
        rough_ICV_mask[...,z] = slice_rough_ICV_mask

    labels = cc3d.connected_components(rough_ICV_mask, connectivity=6)
    stats = cc3d.statistics(labels)
    if len(stats['voxel_counts']) >= 2:
        idx = np.argpartition(stats['voxel_counts'], -2)[-2:]
        idx = idx[np.argsort(stats['voxel_counts'][idx])]
    else:
        idx = [0]
    rough_ICV_mask = labels==idx[0]
    rough_ICV_mask = erosion(rough_ICV_mask, footprint=cube(2))
    rough_ICV_mask = dilation(rough_ICV_mask, footprint=cube(2))

    icv_mask = nib.Nifti1Image(rough_ICV_mask, img.affine, img.header)
    nib.save(icv_mask, output_icv_path)
    ct = nib.Nifti1Image(rough_ICV_mask*CT_ori, img.affine, img.header)
    nib.save(ct, output_ct_path)


def png2nii(png_series_dir, src_nii_path, output_nii_path):
    volumn = []
    src_nii = nib.load(src_nii_path)
    i = 0
    for file in os.listdir(png_series_dir):
        img = Image.open(os.path.join(png_series_dir, file))
        img = np.array(img)
        volumn.append(img)
        i += 1
    volumn = np.array(volumn)
    volumn = np.flip(volumn, axis=1)
    volumn = volumn.transpose(2, 1, 0)
    nii = nib.Nifti1Image(volumn, src_nii.affine, src_nii.header)
    nib.save(nii, output_nii_path)


##########  Main 
if __name__ == '__main__':
    
    data_path = '/media/yingchihlin/4TBHDD/2022_vessel/Data'
    folders = [f for f in listdir(data_path)]
    folders.sort()
                    

    for folder in folders:
        
        # month 202201
        if folder == '202201':
            folder_path = os.path.join(data_path, folder)  
            
            #case 
            cases = [c for c in listdir(folder_path)]
            cases.sort()          

            start = 0
            for i in range(start, len(cases)):
                print(folder, i, '/{}'.format(len(cases)))
                cases_path = os.path.join(folder_path, cases[i], 'CTP_Nii')  
                
                #Build a skull stripping folder
                ss_path = os.path.join(cases_path, 'skull_strip')
                if not os.path.exists(ss_path):
                    os.mkdir(ss_path)
                        
                #Build a skull stripping mask folder
                ssm_path = os.path.join(cases_path, 'skull_strip_mask')
                if not os.path.exists(ssm_path):
                    os.mkdir(ssm_path)
                
                #nii data
                check_path = os.path.join(cases_path, '*.nii')
                nii_list = glob.glob(check_path)
                nii_list.sort()
                
                for nii in tqdm(nii_list):
                    nii_path = nii
                    file_name = nii_path.split('/')[-1]
                    ss_name = 'ss_' + file_name
                    ssm_name = 'ssm_' + file_name
                    output_icv_path = os.path.join(ssm_path, ssm_name)
                    output_ct_path = os.path.join(ss_path, ss_name)

                    skull_strip(nii_path, output_ct_path, output_icv_path)
    


