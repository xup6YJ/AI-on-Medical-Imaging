import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding="same")
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out




class EEGNet(nn.Module):
    def __init__(self, args):
        
        super(EEGNet, self).__init__()
        # self.nb_classes = nb_classes
        self.dropoutRate = args.dropout_rate
        self.activation = args.activation_function
        self.alpha = args.elu_alpha

        
        ##################################################################
        # model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples, 
        #        dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
        #        dropoutType = 'Dropout')
        # kernels, chans, samples = 1, 60, 151


        # nb_classes, Chans = 64, Samples = 128, 
        #      dropoutRate = 0.5, kernLength = 64, F1 = 8, 
        #      D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'

        #  input1   = Input(shape = (Chans, Samples, 1))
        
        # block1       = Conv2D(F1, (1, kernLength), padding = 'same',
        #                             input_shape = (Chans, Samples, 1),
        #                             use_bias = False)(input1)
        # block1       = BatchNormalization()(block1)
        # block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
        #                             depth_multiplier = D,
        #                             depthwise_constraint = max_norm(1.))(block1)

        # block1       = BatchNormalization()(block1)
        # block1       = Activation('elu')(block1)
        # block1       = AveragePooling2D((1, 4))(block1)
        # block1       = dropoutType(dropoutRate)(block1)
        
        # block2       = SeparableConv2D(F2, (1, 16),
        #                             use_bias = False, padding = 'same')(block1)
        # block2       = BatchNormalization()(block2)
        # block2       = Activation('elu')(block2)
        # block2       = AveragePooling2D((1, 8))(block2)
        # block2       = dropoutType(dropoutRate)(block2)
            
        # flatten      = Flatten(name = 'flatten')(block2)
        
        # dense        = Dense(nb_classes, name = 'dense', 
        #                     kernel_constraint = max_norm(norm_rate))(flatten)
        # softmax      = Activation('softmax', name = 'softmax')(dense)
        ##################################################################


        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 51), padding = (0, 25), bias = False)  #(1, 64)
        # self.conv1 = nn.Conv2d(1, self.F1, (1, 64))  #(1, 64)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        #DepthwiseConv2D D * F1 (C, 1)
        '''When groups == in_channels and out_channels == K * in_channels, 
        where K is a positive integer, this operation is also known as a “depthwise convolution”.
        '''
        #16, 1, 2, 1
        self.depthwise_conv1= nn.Conv2d(in_channels=16, out_channels=32, stride = (1,1),
                                        kernel_size=(2, 1), groups=16)
        self.batchnorm2 = nn.BatchNorm2d(32, False)
        self.avgpooling = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)

        # Layer 2
        # self.sep_depthwise = nn.Conv2d(16, self.F2, kernel_size=(1, 16), groups=16, bias=False)
        self.separable_conv = nn.Conv2d(32, 32, kernel_size=(1, 15), stride = (1,1), bias=False, padding=(0,7))
        
        self.batchnorm3 = nn.BatchNorm2d(32, False)
        self.avgpooling2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        # self.fc1 = nn.Linear(384, 1)
        self.fc1 = nn.Linear(in_features=736, out_features=2, bias=True)

        if self.activation == 'relu':
            self.activation_func = nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.activation_func = nn.LeakyReLU()
        elif self.activation == 'elu':
            self.activation_func = nn.ELU(alpha = self.alpha)
        elif self.activation == 'selu':
            self.activation_func = nn.SELU()
        else:
            raise ValueError('Invalid activation function.')
        

    def forward(self, x):
         
        # print('Input x.shape: ', x.shape)  #Input x.shape:  torch.Size([64, 1, 2, 750])

        # Block 1
        x = self.conv1(x)
        # print('self.conv1(x) x.shape: ', x.shape)
        x = self.batchnorm1(x)
        # print('self.batchnorm1(x) x.shape: ', x.shape)
        x = self.depthwise_conv1(x)
        # print('self.depthwise_conv1(x) x.shape: ', x.shape)
        x = self.batchnorm2(x)
        # print('self.batchnorm2(x) x.shape: ', x.shape)
        x = self.activation_func(x)
        # print('F.elu(x) x.shape: ', x.shape)
        x = self.avgpooling(x)
        # print('self.avgpooling(x) x.shape: ', x.shape)
        x = F.dropout(x, self.dropoutRate)
        # print('F.dropout(x, self.dropout) x.shape: ', x.shape)
        
        # Block 2
        x = self.separable_conv(x)
        # print('SeparableConv2d(16, self.F2, kernel_size = (1, 16)) x.shape: ', x.shape)
        x = self.batchnorm3(x)
        # print('self.batchnorm2(x) x.shape: ', x.shape)
        x = self.activation_func(x)
        # print('F.elu(x) x.shape: ', x.shape)

        x = self.avgpooling2(x)
        # print('self.avgpooling2(x) x.shape: ', x.shape)
        x = F.dropout(x, self.dropoutRate)
        # print('F.dropout(x, self.dropout) x.shape: ', x.shape)
        
        # FC Layer
        x = torch.flatten(x, start_dim=1)
        # print('torch.flatten(x) x.shape', x.shape)
        x = self.fc1(x)
        # print('self.fc1(x) x.shape', x.shape)
        x = F.sigmoid(x)
        # print('F.sigmoid(x) x.shape', x.shape)
        return x
    



class EEGNet_2(nn.Module):
    def __init__(self, args):
        super(EEGNet_2, self).__init__()
        self.dropoutRate = args.dropout_rate
        self.alpha = args.elu_alpha
        self.activation = args.activation_function

        if self.activation == 'relu':
            self.activation_func = nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.activation_func = nn.LeakyReLU()
        elif self.activation == 'elu':
            self.activation_func = nn.ELU(alpha = self.alpha)
        elif self.activation == 'selu':
            self.activation_func = nn.SELU()
        else:
            raise ValueError('Invalid activation function.')
    
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activation_func,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=self.dropoutRate)
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activation_func,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=self.dropoutRate)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.flatten(1)
        x = self.classify(x)
        return x