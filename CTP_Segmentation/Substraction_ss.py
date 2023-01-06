

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
import nibabel as nib
import numpy as np
import cv2
import numpy as geek
import cc3d
import logging
from tqdm import tqdm
import nibabel as nib
from skimage.morphology import erosion, dilation, cube
from scipy.ndimage import binary_fill_holes

def show_all_img(result, columns = None, rows = 1, save_img = False, save_path = None):

    if columns is None:
        columns = len(result)
    fig = plt.figure(figsize=(15, 15))

    for i, img_n in enumerate(result):
        ax = fig.add_subplot(rows, columns, i+1)
        ax.title.set_text(img_n)
        # ax.set_title(i)  
        img = result[img_n]
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            
    if  save_img:
        plt.savefig(save_path)
        plt.close()

    
    #else:
        #plt.show()
    
def skull_strip(nii_path, output_ct_path, output_icv_path):
    intensity_thres = 100
    img = nib.load(nii_path)
    ct = img.get_fdata()
    ct_ori = np.array(ct)
    

    # if erase_outside:
    #     bone_region = ct >= intensity_thres
    #     outside_bone = np.zeros(ct.shape)
    #     outside_bone[bone_region] = 1
    #     outside_bone = dilation(outside_bone, footprint=cube(4))
    #     for z in range(outside_bone.shape[-1]):
    #         outside_bone[:,:,z] = binary_fill_holes(outside_bone[:,:,z])
    #     outside_bone = 1 - outside_bone

    
    ct[ct<0] = 0
    ct[ct>=intensity_thres] = 0

    ct_ori[ct_ori<0] = 0
    ct_ori[ct_ori>intensity_thres] = intensity_thres
    ct_ori = ct_ori / intensity_thres

    # if erase_outside:
    #     ct[outside_bone==1] = 0

    mask = np.array(ct)
    mask[mask>0] = 1
    
    cc = cc3d.connected_components(mask, connectivity=6)
    stats = cc3d.statistics(cc)
    idx = np.arange(len(stats['voxel_counts']))
    idx = idx[np.argsort(stats['voxel_counts'])[::-1]]
    tissue_mask = cc == idx[1]
    tissue_mask = erosion(tissue_mask, footprint=cube(3))
    tissue_mask = dilation(tissue_mask, footprint=cube(3))

    # rough_icv_mask = tissue_mask
    rough_icv_mask = np.zeros(ct.shape)
    for z in range(ct.shape[-1]):
        slice_tissue_mask = tissue_mask[...,z]
        if slice_tissue_mask.sum() == 0:
            continue
        cc = cc3d.connected_components(slice_tissue_mask, connectivity=4)
        stats = cc3d.statistics(cc)
        idx = np.arange(len(stats['voxel_counts']))
        idx = idx[np.argsort(stats['voxel_counts'])[::-1]]
        slice_rough_icv_mask = np.zeros(slice_tissue_mask.shape)
        
        for i in idx:
            if stats['voxel_counts'][i]>150000 or stats['voxel_counts'][i]<1500:
                continue
            component = cc == i
            x, y = np.where(component)
            y_mid = (y.min()+y.max()) // 2
            x_mid = (x.min()+x.max()) // 2
            component_fill = binary_fill_holes(component)
            diff = component_fill - component.astype(np.uint8)
            center_area = component[x_mid-3:x_mid+4,y_mid-3:y_mid+4].sum()
            # if center_area and diff.sum()<component_fill.sum()*0.4:
            #     slice_rough_icv_mask = component
            #     break
            if diff.sum()<component_fill.sum()*0.4:
                slice_rough_icv_mask = component
                break
        slice_rough_icv_mask = binary_fill_holes(slice_rough_icv_mask)
        rough_icv_mask[...,z] = slice_rough_icv_mask

    cc = cc3d.connected_components(rough_icv_mask, connectivity=6)
    stats = cc3d.statistics(cc)
    idx = np.arange(len(stats['voxel_counts']))
    idx = idx[np.argsort(stats['voxel_counts'])[::-1]]
    rough_icv_mask = cc == idx[1]
    rough_icv_mask = erosion(rough_icv_mask, footprint=cube(3))
    rough_icv_mask = dilation(rough_icv_mask, footprint=cube(3))

    # if check_area:
    #     meet_maximum = False
    #     prev_area = -1
    #     for z in range(rough_icv_mask.shape[-1]-1, -1, -1):
    #         area = rough_icv_mask[:,:,z].sum()
    #         if area < prev_area:
    #             meet_maximum = True
    #             prev_area = area
    #         elif area > prev_area:
    #             if meet_maximum:
    #                 rough_icv_mask[:,:,z] = 0
    #                 prev_area = 0
    #             else:
    #                 prev_area = area

    icv_mask = nib.Nifti1Image(rough_icv_mask, img.affine, img.header)
    nib.save(icv_mask, output_icv_path)
    ct = nib.Nifti1Image(rough_icv_mask*ct_ori, img.affine, img.header)
    nib.save(ct, output_ct_path)

    
###############################################################################
data_path = '/media/yingchihlin/4TBHDD/2022_vessel/Data'
folders = [f for f in listdir(data_path)]
folders.sort()
# for folder in folders:
    
# month 202201

folder_path = os.path.join(data_path, folders[0])  

#case 
cases = [c for c in listdir(folder_path)]
cases.sort()

for case in cases:
    print( 'Case: ', case)

    path = os.path.join(folder_path, case, 'CTP_Nii') 
    
    #subtraction file path
    subtraction_file_path = os.path.join(path, 'Subtraction_ss')
    if not os.path.exists(subtraction_file_path):
        os.mkdir(subtraction_file_path)
    
    ss_path = os.path.join(path, 'skull_strip') 
    check_path = os.path.join(path, '*.nii')
    nii_list = glob.glob(check_path)
    nii_list.sort()
    
    #Read
    ctp_0 = nib.load(os.path.join(ss_path, 'ss_CTP_0.nii')).get_fdata()
    ctp_0.shape
    
    times = 1
    while times <= 22:
        
        print('Substracting {}-0'.format(times))
        ctp_1 = nib.load(os.path.join(ss_path, 'ss_CTP_{}.nii'.format(times))).get_fdata()
        assert ctp_0.shape == ctp_1.shape
        
        #Build a time series folder
        name = str(times) + '-0'
        sub_path = os.path.join(subtraction_file_path, name)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
            
        slice = ctp_0.shape[2]
        for i in tqdm(range(slice)):
        
            #times 0 
            o_ctp_0 = ctp_0[:,:,i]
            # o_ncct[o_ncct<0] = 0
            # mask_ncct = o_ncct>120  #Bone
            # mask_ncct = mask_ncct.astype(int)
            # o_mask_ncct = np.clip(mask_ncct*o_ncct, 0, 1)
            
            #times > 0
            o_ctp_1 = ctp_1[:,:,i]
            # mask_cta = o_cta>120  #Bone
            # mask_cta = mask_cta.astype(int)
            # o_mask_cta = np.clip(mask_cta*o_cta, 0, 1)
            
            #Substraction
            vessel = geek.subtract(o_ctp_1, o_ctp_0) 
            vessel = np.clip(vessel, 0, 255)
            
            
            ###################################
            result = {}
            result['Time = 0'] = o_ctp_0
            name = 'Times = {}'.format(times)
            result[name] = o_ctp_1
            
            # result['NCCT_bone'] = o_ncct*mask_ncct
            # result['NCCT_bone_mask'] = mask_ncct
            
            # result['CTA_bone_vessel'] = mask_cta*o_cta
            # result['CTA_bone_vessel_mask'] = mask_cta
            
            result['Substraction'] = vessel
            
            fig_name = 'Sub_ss' + str(i) + '.jpg'
            save_path = os.path.join(sub_path, fig_name)
            show_all_img(result, columns = 3, rows = 1, save_img = True, save_path = save_path)
            
        times +=1
