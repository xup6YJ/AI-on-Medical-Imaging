


import pandas as pd
import time
import re
import numpy as np
import scipy.io
import requests
import math
from glob import glob

import os
from os import listdir
from os.path import isfile, isdir, join, splitext
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold, KFold
from datetime import datetime
import glob
import shutil
import random
from torch.optim.swa_utils import *
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 106
np.random.seed(seed)

'''
Train infection ratio 0.009125585644785437 (14)
Valid infection ratio 0.00891329907723825 (2)
Test infection ratio 0.009681017562080525 (4)
'''

# Let's say we want to split the data in 80:10:10 for train:valid:test dataset
df = pd.read_csv('/home/yclin/Documents/MedAI/Final/archive/input/covid19-ct-scans/metadata_2.csv')

train_size=0.8

X = df.drop(columns = ['infection_mask']).copy()
y = df['infection_mask']
y.columns = ['infection_mask']
# In the first step we will split the data in training and remaining dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_size)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
ac_train_size = 0.9
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, train_size=ac_train_size)

print(X_train.shape), print(y_train.shape)
print('Train infection ratio', X_train['Ratio'].mean())
print(X_valid.shape), print(y_valid.shape)
print('Valid infection ratio', X_valid['Ratio'].mean())
print(X_test.shape), print(y_test.shape)
print('Test infection ratio', X_test['Ratio'].mean())



cur_path = os.getcwd()
data_path = os.path.join(cur_path, 'data')
ct_path = os.path.join(data_path, 'CT_data')
lung_path = os.path.join(data_path, 'lung_data')
infec_path = os.path.join(data_path, 'infection_data')

train_path = os.path.join(data_path, 'train')
if not os.path.exists(train_path):
    os.makedirs(train_path)
valid_path = os.path.join(data_path, 'valid')
if not os.path.exists(valid_path):
    os.makedirs(valid_path)
test_path = os.path.join(data_path, 'test')
if not os.path.exists(test_path):
    os.makedirs(test_path)

train_ct_path = os.path.join(train_path, 'CT')
if not os.path.exists(train_ct_path):
    os.makedirs(train_ct_path)
train_lung_path = os.path.join(train_path, 'Lung')
if not os.path.exists(train_lung_path):
    os.makedirs(train_lung_path)
train_infec_path = os.path.join(train_path, 'Infection')
if not os.path.exists(train_infec_path):
    os.makedirs(train_infec_path)

valid_ct_path = os.path.join(valid_path, 'CT')
if not os.path.exists(valid_ct_path):
    os.makedirs(valid_ct_path)
valid_lung_path = os.path.join(valid_path, 'Lung')
if not os.path.exists(valid_lung_path):
    os.makedirs(valid_lung_path)
valid_infec_path = os.path.join(valid_path, 'Infection')
if not os.path.exists(valid_infec_path):
    os.makedirs(valid_infec_path)

test_ct_path = os.path.join(test_path, 'CT')
if not os.path.exists(test_ct_path):
    os.makedirs(test_ct_path)
test_lung_path = os.path.join(test_path, 'Lung')
if not os.path.exists(test_lung_path):
    os.makedirs(test_lung_path)
test_infec_path = os.path.join(test_path, 'Infection')
if not os.path.exists(test_infec_path):
    os.makedirs(test_infec_path)

#Train
print('Copying training data')
indexes = X_train.index
for ind in indexes:
    #CT
    name = 'p_' + os.path.split(X_train.loc[ind,'ct_scan'])[-1]
    path = os.path.join(ct_path, name)
    #Copy
    source = path
    destination = os.path.join(train_ct_path, name)
    dest = shutil.copyfile(source, destination)

    #Lung
    name = 'p_' + os.path.split(X_train.loc[ind,'lung_mask'])[-1]
    path = os.path.join(lung_path, name)
    #Copy
    source = path
    destination = os.path.join(train_lung_path, name)
    dest = shutil.copyfile(source, destination)

    #Infection
    name = 'p_' + os.path.split(y_train.loc[ind])[-1]
    path = os.path.join(infec_path, name)
    #Copy
    source = path
    destination = os.path.join(train_infec_path, name)
    dest = shutil.copyfile(source, destination)

#Validation
print('Copying validation data')
indexes = X_valid.index
for ind in indexes:
    #CT
    name = 'p_' + os.path.split(X_valid.loc[ind,'ct_scan'])[-1]
    path = os.path.join(ct_path, name)
    #Copy
    source = path
    destination = os.path.join(valid_ct_path, name)
    dest = shutil.copyfile(source, destination)

    #Lung
    name = 'p_' + os.path.split(X_valid.loc[ind,'lung_mask'])[-1]
    path = os.path.join(lung_path, name)
    #Copy
    source = path
    destination = os.path.join(valid_lung_path, name)
    dest = shutil.copyfile(source, destination)

    #Infection
    name = 'p_' + os.path.split(y_valid.loc[ind])[-1]
    path = os.path.join(infec_path, name)
    #Copy
    source = path
    destination = os.path.join(valid_infec_path, name)
    dest = shutil.copyfile(source, destination)


#Test
print('Copying testing data')
indexes = X_test.index
for ind in indexes:
    #CT
    name = 'p_' + os.path.split(X_test.loc[ind,'ct_scan'])[-1]
    path = os.path.join(ct_path, name)
    #Copy
    source = path
    destination = os.path.join(test_ct_path, name)
    dest = shutil.copyfile(source, destination)

    #Lung
    name = 'p_' + os.path.split(X_test.loc[ind,'lung_mask'])[-1]
    path = os.path.join(lung_path, name)
    #Copy
    source = path
    destination = os.path.join(test_lung_path, name)
    dest = shutil.copyfile(source, destination)

    #Infection
    name = 'p_' + os.path.split(y_test.loc[ind])[-1]
    path = os.path.join(infec_path, name)
    #Copy
    source = path
    destination = os.path.join(test_infec_path, name)
    dest = shutil.copyfile(source, destination)
