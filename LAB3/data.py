
import glob
import pandas  as pd
import numpy   as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

#Open folder in input
def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)

def plot_sample(array_list, color_map = 'nipy_spectral'):
    '''
    Plots and a slice with all available annotations
    '''
    fig = plt.figure(figsize=(18,15))

    plt.subplot(1,4,1)
    plt.imshow(array_list[0], cmap='bone')
    plt.title('Original Image')

    plt.subplot(1,4,2)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Lung Mask')

    plt.subplot(1,4,3)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[2], alpha=0.5, cmap=color_map)
    plt.title('Infection Mask')

    plt.subplot(1,4,4)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[3], alpha=0.5, cmap=color_map)
    plt.title('Lung and Infection Mask')

    plt.show()

# raw_data = pd.read_csv('/home/yclin/Documents/MedAI/Final/archive/metadata.csv')   
raw_data = pd.read_csv('../input/covid19-ct-scans/metadata.csv')
raw_data.sample(5)

# Read sample
shape_list = []
for i in range(len(raw_data)):
    sample_ct   = read_nii(raw_data.loc[i,'ct_scan'])
    sample_lung = read_nii(raw_data.loc[i,'lung_mask'])
    sample_infe = read_nii(raw_data.loc[i,'infection_mask'])
    sample_all  = read_nii(raw_data.loc[i,'lung_and_infection_mask'])

    # Examine Shape
    shape_list.append(sample_ct.shape)
    sample_ct.shape

raw_data['Shape'] = shape_list
raw_data.to_csv('../input/covid19-ct-scans/metadata_shape.csv', index = False)
# Examine one slice of a ct scan and its annotations

plot_sample([sample_ct[...,120], sample_lung[...,120], sample_infe[...,120], sample_all[...,120]])