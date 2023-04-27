


import glob
import pandas  as pd
import numpy   as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


#Open folder in input
def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    # array   = np.rot90(np.array(array))
    affine = ct_scan.affine
    header = ct_scan.header
    return(array, affine, header)

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]
    z = array.shape[2]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant')


def calculate_slice(label):
    # label = np.concatenate(self.label_list)
    label = label.flatten()
    normal_data_num = len( np.argwhere(label == 0))
    detect_data_num = len( np.argwhere(label == 1))
    data_num = normal_data_num + detect_data_num
    affect_ratio = float(detect_data_num) / float(data_num)
    # repr_str = "TorchDataset\n\t%d volumes,\t%d ROIs\n\t%d normal ROIs,\t%d defect ROIs (%.2f%% affected)" %(len(self),data_num,normal_data_num,detect_data_num,affect_ratio*100)
    repr_str = "COVIDDataset\n\t volumes,\t%d ROIs\n\t%d normal ROIs,\t%d defect ROIs (%.2f%% affected)" %(data_num,normal_data_num,detect_data_num,affect_ratio*100)
    print(repr_str)
    
    return affect_ratio


if __name__ == '__main__':

    cur_path = os.getcwd()
    data_path = os.path.join(cur_path, 'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    ct_path = os.path.join(data_path, 'CT_data')
    if not os.path.exists(ct_path):
        os.makedirs(ct_path)

    lung_path = os.path.join(data_path, 'lung_data')
    if not os.path.exists(lung_path):
        os.makedirs(lung_path)

    infec_path = os.path.join(data_path, 'infection_data')
    if not os.path.exists(infec_path):
        os.makedirs(infec_path)

    # raw_data = pd.read_csv('/home/yclin/Documents/MedAI/Final/archive/metadata.csv')   
    raw_data = pd.read_csv('../input/covid19-ct-scans/metadata_shape.csv')
    raw_data.sample(5)

    # Read sample
    shape_list = []
    ratio_list = []
    for i in tqdm(range(len(raw_data))):
        sample_ct, affine, header   = read_nii(raw_data.loc[i,'ct_scan'])
        sample_lung, _, _ = read_nii(raw_data.loc[i,'lung_mask'])
        sample_infe, _, _ = read_nii(raw_data.loc[i,'infection_mask'])
        # sample_all  = read_nii(raw_data.loc[i,'lung_and_infection_mask'])

        #Examine Shape
        # shape_list.append(sample_ct.shape)
        # sample_ct.shape

        shape = sample_ct.shape
        shape_h_w = shape[:2]

        if shape_h_w != (630, 630):
            sample_ct = padding(sample_ct, 630, 630)  #resize h and w only do padding
            sample_lung = padding(sample_lung, 630, 630)
            sample_infe = padding(sample_infe, 630, 630)

        ratio = calculate_slice(sample_infe)
        ratio_list.append(ratio)

        #Save data
        ct_nii = nib.Nifti1Image(sample_ct*sample_lung, affine, header)
        lung_nii = nib.Nifti1Image(sample_lung, affine, header)
        infe_nii = nib.Nifti1Image(sample_infe, affine, header)

        name = 'p_' + os.path.split(raw_data.loc[i,'ct_scan'])[-1]
        path = os.path.join(ct_path, name)
        nib.save(ct_nii, path)

        name = 'p_' + os.path.split(raw_data.loc[i,'lung_mask'])[-1]
        path = os.path.join(lung_path, name)
        nib.save(lung_nii, path)

        name = 'p_' + os.path.split(raw_data.loc[i,'infection_mask'])[-1]
        path = os.path.join(infec_path, name)
        nib.save(infe_nii, path)


# raw_data['Shape'] = shape_list
# raw_data['Ratio'] = ratio_list
# raw_data.to_csv('../input/covid19-ct-scans/metadata_2.csv', index = False)