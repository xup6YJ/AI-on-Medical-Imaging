

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
data_path = '/media/yingchihlin/4TBHDD/2022_vessel/Data'
folders = [f for f in listdir(data_path)]
folders.sort()
p_list = []

for folder in folders:
    
    # month 202201
    if folder == '202210':
        folder_path = os.path.join(data_path, folder)  
        
        #case 
        cases = [c for c in listdir(folder_path)]
        cases.sort()
        
        if folder == '202210':
            start = 11
        else:
            start = 0
        
        for i in tqdm(range(start, len(cases))):
            cases_path = os.path.join(folder_path, cases[i])  
            
            #series
            series = [s for s in listdir(cases_path)]
            
            #preserve series data
            if 'desktop.ini' in series:
                series.remove('desktop.ini')
            
            n_data = []
            for s in series:
                series_path = os.path.join(cases_path, s)  
                data = [d for d in listdir(series_path)]
                ndata = len(data)
                n_data.append(ndata)
            
            #CTP File
            result = np.where(n_data == np.max(n_data))[0][0]
            ctp_file = series[result]
            ctp_file_path = os.path.join(cases_path, ctp_file)  
            
            check_path = os.path.join(ctp_file_path, '*.dcm')
            dcm_list = glob.glob(check_path)
            dcm_list.sort()
            
            #Build a folder
            # pic_path = os.path.join(cases_path, 'JPG')
            # if not os.path.exists(pic_path):
                # os.mkdir(pic_path)
            
            #Check 160 slice        
            # for i in range(480):
            #     data = dcm_list[i]
            #     ds0 = pydicom.dcmread(data)
                
            #     uid_dcm_df = pd.DataFrame(columns=['ID', 'Header', 'Modality', 'Age', 'BirthDate', 'Sex', 'Date', 'SliceThickness'])
            #     #header
            #     header0 = ds0.SeriesDescription 
            #     print(header0)
                
            #     # ds0.Modality
            #     # ds0.PatientAge
            #     # ds0.PatientBirthDate
            #     # ds0.PatientSex
            #     # ds0.PerformedProcedureStepStartDate
            #     # ds0.SliceThickness
            #     # ds0.pixel_array
                
            #     # Plot
            #     # plt.imshow(ds0.pixel_array, cmap='gray')
            #     # plt.show()
            #     dcm_data = ds0.pixel_array
            #     cv2.imwrite(os.path.join(pic_path, '{}.jpg'.format(i)), dcm_data)
            
            
            #####  Dicom to Nifty
            log_level = logging.INFO
            logging.basicConfig(level=logging.ERROR, format='%(levelname)s %(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
            settings.disable_validate_orthogonal()
            settings.disable_validate_slice_increment()
            
            #Build a Nifty folder
            nii_path = os.path.join(cases_path, 'CTP_Nii')
            if not os.path.exists(nii_path):
                    os.mkdir(nii_path)
                    
            #Build a time series folder
            time_path = os.path.join(cases_path, 'series_time')
            if not os.path.exists(time_path):
                    os.mkdir(time_path)
            
            dcm_list.sort()
            
            times = 0
            
            while times <= 22: #total = 23
                print(folder, 'No.', times, 'series...')
                time_dcm = dcm_list[times*160:(times+1)*160]  #160 a subject
                time_dcm.sort()
                
                #Build a time series folder
                time_series_path = os.path.join(time_path, str(times))
                if not os.path.exists(time_series_path):
                    os.mkdir(time_series_path)
                
                #put a time seires in a folder
                for t_path in time_dcm:
                    # Source path NCCT
                    source = t_path
                    # Destination path
                    file_name = t_path.split('/')[-1]
                    destination = os.path.join(time_series_path, file_name)
                    shutil.copy(source, destination)
                
                ###Remove Problem Data
                problem_list = ['/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/5/instance_33110.dcm',
                                '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/5/instance_33122.dcm',
                                '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/6/instance_33150.dcm',
                                '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/7/instance_33342.dcm',
                                 '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/7/instance_33390.dcm',
                                 '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/7/instance_33452.dcm',
                                 '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/10/instance_33812.dcm',
                                  '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/10/instance_33841.dcm',
                                  '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/10/instance_33850.dcm',
                                  '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/11/instance_34046.dcm',
                                  '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/12/instance_33110.dcm',
                                  '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/12/instance_33122.dcm',
                                  '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/12/instance_34210.dcm',
                                   '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/12/instance_34232.dcm',
                                   '/media/yingchihlin/4TBHDD/2022_vessel/Data/202210/262/series_time/12/instance_34238.dcm']
                
                for p_dcm in problem_list:
                    if os.path.exists(p_dcm):
                        os.remove(p_dcm)
                
                check_path = os.path.join(time_series_path, '*.dcm')
                dc_list = glob.glob(check_path)
                dc_list.sort()
                
                for d in dc_list:
                    ds0 = pydicom.dcmread(d)
                    try:
                        d_data = ds0.pixel_array
                    except ValueError:
                        p_list.append(d)
                        
                for p_dcm in p_list:
                    if os.path.exists(p_dcm):
                        os.remove(p_dcm)
                
                #Directory
                # dcm_series_dir = os.path.join('dicom', subject, 'DWI')
                # mask_series_dir = os.path.join('mask', subject)
                # output_dir = os.path.join('nii', subject)
                # os.makedirs(output_dir, exist_ok=True)
                # dcm2nii_path = os.path.join(output_dir, 'dcm2nii.nii')
                # reorient_path = os.path.join(output_dir, 'reorient.nii')
                # reorient_ss_path = os.path.join(output_dir, 'reorient_ss.nii')
                # reorient_icv_path = os.path.join(output_dir, 'reorient_icv.nii')
                # mask_path = os.path.join(output_dir, 'mask.nii')
                
                #CTP
                dcm2nii_path = os.path.join(nii_path, 'CTP_{}.nii'.format(times))
                #os.remove(dcm2nii_path)
                
                #shutil
                logging.log(log_level, 'NCCT Dicom serires to NIfTI...')
                dcm2nii(time_series_path, dcm2nii_path)
                
                times +=1
    


