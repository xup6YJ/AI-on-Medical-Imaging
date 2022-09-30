import os
from os import listdir
from os.path import isfile, join
import os
from xml.etree.ElementInclude import include
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import glob
import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging

import torchio as tio
from torchio.data import SubjectsDataset
from torchio.transforms.spatial_transform import SpatialTransform
from torchio.transforms.intensity_transform import IntensityTransform
import torchvision.transforms.functional as tF

from util.logconf import logging
from utils import chwd2cdhw, cdhw2chwd, flip_z

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class CopyAffine(IntensityTransform):
    def __init__(self, **kwargs):
        super(CopyAffine, self).__init__(**kwargs)

    def apply_transform(self, subject):
        resample = tio.CopyAffine('image')
        subject = resample(subject)
        return subject


class RandomAffine(SpatialTransform):
    def __init__(self, degrees=0, translate=None, scale=None, **kwargs):
        super(RandomAffine, self).__init__(**kwargs)
        self.degrees = degrees
        self.translate = 0 if translate is None else translate
        self.scale = 0 if scale is None else scale

    def apply_transform(self, subject):
        angle = np.random.randint(-self.degrees, self.degrees)
        translate = np.random.rand(2) * self.translate * 256
        scale = 1 - (np.random.rand()*2-1) * self.scale
        for image in self.get_images(subject):
            new_data = tF.affine(
                            chwd2cdhw(image.data),
                            angle=angle, translate=translate.tolist(),
                            scale=scale, shear=[0, 0])
            new_data = cdhw2chwd(new_data)
            image.set_data(new_data)
        return subject


class SampleProbabilityMap(IntensityTransform):
    def __init__(self, icv_weight=0, **kwargs):
        super(SampleProbabilityMap, self).__init__(**kwargs)
        self.icv_weight = icv_weight

    def apply_transform(self, subject):
        has_roi = (subject['label'][tio.DATA].sum(dim=[1, 2])>0).type(torch.float32)
        has_brain = (subject['icv'][tio.DATA].sum(dim=[1, 2])>0).type(torch.float32)
        
        # print('Data shape:', subject['icv'][tio.DATA].shape)
        # print('has_brain: ', has_brain.shape)
        # print('has_roi: ', has_roi.shape)
        # print('icv_weight: ', self.icv_weight)

        prob_map = subject['icv'][tio.DATA] * (self.icv_weight*has_brain+(1-self.icv_weight)*has_roi)

        subject['prob_map'].set_data(prob_map)
        return subject


#Dataset

class CTDataset(SubjectsDataset):
    def __init__(
        self, validation = False, flip_p=0.5, trans_range=0.1, resize_range=0.1, 
        rotate_range=20, noise_mm=10, no_noisy=False, flip_image=False, curriculum=True, **kwargs):

        if not validation:
            data_path = '/home/yingchihlin/Documents/Code/AISD/train'
        else:
            data_path = '/home/yingchihlin/Documents/Code/AISD/validation'

        id_list = os.listdir(data_path)
        id_list = sorted(id_list)
        # print(id_list)
        image_paths = [glob.glob(data_path + '/{}/CT.nii.gz'.format(uid))[0] for uid in id_list]

        # if not validation:
        #     image_paths = [glob.glob('/home/yingchihlin/Documents/Code/AISD/train/{}/CT.nii.gz'.format(uid))[0] for uid in id_list]
        # else:
        #     image_paths = [glob.glob('/home/yingchihlin/Documents/Code/AISD/validation/{}/CT.nii.gz'.format(uid))[0] for uid in id_list]

        # flip_p = 0 if flip_p is None else flip_p
        # image_dir = join(data_dir, 'images')
        # image_paths = sorted(glob.glob(join(image_dir, '*.nii.gz')))
        if flip_image:
            image_flip_paths = [path.replace('CT', 'images_flip') for path in image_paths]
        else:
            image_flip_paths = image_paths
        roi_paths = [path.replace('CT', 'mask') for path in image_paths]
        icv_paths = [path.replace('CT', 'ICV') for path in image_paths]

        image_num = len(image_paths)
        # file_names = [os.path.split(x)[1] for x in image_paths]
        subj_names = [os.path.split(image_paths[x])[0][-7:] for x in range(len(image_paths))]

        if not validation:
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    # image_flip=tio.ScalarImage(image_flip_paths[i]),
                    label=tio.ScalarImage(roi_paths[i]),
                    # dist_map=tio.ScalarImage(roi_paths[i]),
                    prob_map=tio.ScalarImage(icv_paths[i]),
                    icv=tio.ScalarImage(icv_paths[i]),
                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]
        elif validation:
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    # image_flip=tio.ScalarImage(image_flip_paths[i]),
                    label=tio.ScalarImage(roi_paths[i]),
                    # dist_map=tio.ScalarImage(roi_paths[i]),
                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]
        else:
            subjects = [
                tio.Subject(
                    image=tio.ScalarImage(image_paths[i]),
                    # image_flip=tio.ScalarImage(image_flip_paths[i]),
                    image_path=image_paths[i],
                    name=subj_names[i])
                for i in range(image_num)
            ]

        transform = []

        # #???
        if not validation:
            transform += [
                tio.transforms.RandomFlip(axes='LR', flip_probability=flip_p),
                RandomAffine(rotate_range, trans_range, resize_range),  ##
            ]
        #     if not no_noisy:
        #         transform.append(NoisyLabel(noise_mm=noise_mm, include=['label']))  ##
            if curriculum:
                transform.append(tio.transforms.Pad((64, 64, 4)))
                transform.append(SampleProbabilityMap(icv_weight=0, include=['prob_map']))  ##

        # if validation != None:
        #     transform.append(DistanceMap(include=['dist_map']))  ##

        # transform.append(CopyAffine(include=['label', 'icv']))
        transform += [
            tio.ToCanonical(), tio.Resample('image')
        ]
        transform = tio.transforms.Compose(transform)
        super(CTDataset, self).__init__(subjects=subjects, transform=transform, **kwargs)  ##
        
        
# dataset for 52 testing data
class AISTestDataset(SubjectsDataset):
    '''
    NCCT: `data_dir`/images/.*nii.gz
    flipped NCCT: `data_dir`/images_flip/.*nii.gz
    ROI: `data_dir`/masks/.*nii.gz
    '''
    def __init__(self, **kwargs):

        test_data_path = '/home/yingchihlin/Documents/Code/AISD/AISD.test'
        id_list = os.listdir(test_data_path)

        ##############################################
        id_list = sorted(id_list)
        # print(id_list)
        image_paths = [glob.glob('/home/yingchihlin/Documents/Code/AISD/AISD.test/{}/CT.nii.gz'.format(uid))[0] for uid in id_list]
        image_num = len(image_paths)

        subj_names = [os.path.split(image_paths[x])[0][-7:] for x in range(image_num)]

        # image_flip_paths = [x.replace('images', 'images_flip') for x in image_paths]
        roi_paths = [x.replace('CT', 'mask') for x in image_paths]

        subjects = [
            tio.Subject(
                image=tio.ScalarImage(image_paths[i]),
                # image_flip=tio.ScalarImage(image_flip_paths[i]),
                label=tio.ScalarImage(roi_paths[i]),
                dist_map=tio.ScalarImage(roi_paths[i]),
                image_path=image_paths[i],
                name=subj_names[i])
            for i in range(image_num)
        ]

        transform = []

        transform = tio.transforms.Compose(transform)
        super(AISTestDataset, self).__init__(subjects=subjects, transform=transform, **kwargs)  ##