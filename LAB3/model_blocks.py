
import torch.nn as nn
import torchvision.models
import torch
import torch.nn.functional as F
from torchsummary import summary

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(ConvRelu, self).__init__()

        self.convrelu = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride, padding=padding),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return(self.convrelu(x))
    
class ConvReluDrop(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(ConvReluDrop, self).__init__()

        self.convrelu = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride, padding=padding),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return(self.convrelu(x))
    

class ConvSigmoid(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
    super(ConvSigmoid, self).__init__()

    self.convsigmoid = nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm3d(num_features=out_channels),
        nn.Sigmoid()
    )

  def forward(self, x):
    return(self.convsigmoid(x))


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1) #Channel  #Check
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels) #Check
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()

        # Global average pooling
        y = self.avg_pool(x).view(batch_size, channels)

        # Squeeze
        y = self.relu(self.fc1(y))

        # Excite
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1, 1)

        # Scale
        # x = x * y.expand_as(x)  
        output_tensor = torch.mul(x, y)  #Check
        # print ('SE block', 'output shape: ', output_tensor.shape) 

        return output_tensor


class CEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(CEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1) #Channel  #Check
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, 1) #Check
        self.fc3 = nn.Linear(1, in_channels) #Check
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()

        # Global average pooling
        y = self.avg_pool(x).view(batch_size, channels)

        # Squeeze
        y = self.relu(self.fc1(y))

        # Excite
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = self.fc3(y).view(batch_size, channels, 1, 1, 1)

        # Scale
        # x = x * y.expand_as(x)  
        output_tensor = torch.mul(x, y)  #Check
        # print ('SE block', 'output shape: ', output_tensor.shape) 

        return output_tensor
    

class Spatial_block(nn.Module):
    def __init__(self, in_channels):
        super(Spatial_block, self).__init__()

        self.conv1 = ConvRelu(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, padding = 1)
        self.conv2 = ConvSigmoid(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        output_tensor = torch.mul(x, x2)  #Check

        # print ('Spatial block', 'output shape: ', output_tensor.shape) 

        return(output_tensor) #element-wise multiplication

class FVBlock(nn.Module):
    def __init__(self, in_channels):
        super(FVBlock, self).__init__()

        self.in_channels = in_channels
        self.channel_att = SEBlock(self.in_channels, reduction_ratio=2)
        self.spatial_att = Spatial_block(self.in_channels)
        self.conv1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=int(self.in_channels*3), out_channels=self.in_channels, kernel_size=3, padding=1)
        # self.conv2 = ConvRelu(in_channels = int(self.in_channels*3), out_channels = self.in_channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_c = self.channel_att(x_1)
        x_s = self.spatial_att(x_1)
        att = torch.cat((x_1, x_c, x_s), dim=1) #check
        # print ('attention concat', 'output shape: ', att.shape) 
        x_a = self.conv2(att)
        o = x+x_a
        # print ('FV Block', 'output shape: ', o.shape) 
        
        return o

class PASPP_block(nn.Module):
    def __init__(self, in_channels):
        super(PASPP_block, self).__init__()

        self.cd4 = int(in_channels/4)
        self.cd2 = int(in_channels/2)
        self.conv1 = ConvRelu(in_channels = in_channels, out_channels = self.cd4, kernel_size = 1, padding = 0)
        self.conv2 = ConvRelu(in_channels = self.cd2, out_channels = self.cd2, kernel_size = 1, padding = 0)
        self.conv3 = ConvRelu(in_channels = in_channels, out_channels = in_channels, kernel_size = 1, padding = 0)
        self.conv_dil1 = nn.Conv3d(in_channels = self.cd4, out_channels = self.cd4, kernel_size = 3, padding = 1, dilation=1)
        self.conv_dil2 = nn.Conv3d(in_channels = self.cd4, out_channels = self.cd4, kernel_size = 3, padding = 2, dilation=2)
        self.conv_dil4 = nn.Conv3d(in_channels = self.cd4, out_channels = self.cd4, kernel_size = 3, padding = 4, dilation=4)
        self.conv_dil8 = nn.Conv3d(in_channels = self.cd4, out_channels = self.cd4, kernel_size = 3, padding = 8, dilation=8)
        

    def forward(self, x):

        #1b
        x1 = self.conv1(x)
        x2 = self.conv1(x)
        x12 = x1 + x2
        x1 = self.conv_dil1(x1) + x12
        x2 = self.conv_dil2(x2) + x12
        xc12 = torch.cat((x1, x2), dim = 1)
        o1 = self.conv2(xc12)

        #2b
        x3 = self.conv1(x)
        x4 = self.conv1(x)
        x34 = x3 + x4
        x3 = self.conv_dil4(x3) + x34
        x4 = self.conv_dil8(x2) + x34
        xc34 = torch.cat((x3, x4), dim = 1)
        o2 = self.conv2(xc34)

        output_tensor = self.conv3(torch.cat((o1, o2), dim = 1)) #Check
        # print ('PASPP block', 'output shape: ', output_tensor.shape) 

        return(output_tensor) #element-wise multiplication
    
class deconvblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1):
        super(deconvblock, self).__init__()

        # self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        #                                             stride=stride, padding=padding, output_padding = output_padding)

        k = (2, 2, 1)
        s = (2, 2, 1)

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, output_padding = output_padding),
            # nn.ConvTranspose3d(in_channels, out_channels, kernel_size=k, stride=s, padding=0),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print('Deconv shape: ', self.deconv(x).shape)
        return(self.deconv(x))


class ResConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, n=2):
        super(ResConvBlock, self).__init__()

        kx = ky = kz = 3
        px = py = pz = 1
        epsilon = 1e-7

        norm_mode = 'bn'
        num_groups = None
        ch_per_group = 16


        k = (kx, ky, kz)
        p = (px, py, pz)
        self.conv_1x1 = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        layers = []
        for _ in range(n):
            layers += [
                nn.Conv3d(out_channel, out_channel, kernel_size=k, stride=1, padding=p, bias=bias),
                Norm(out_channel),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x2 = self.conv(x1)
        return x1 + x2
    

class Norm(nn.Module):
    def __init__(self, channel):
        super(Norm, self).__init__()

        epsilon = 1e-7
        norm_mode = 'bn'
        num_groups = None
        ch_per_group = 16

        if norm_mode == 'bn':
            self.norm = nn.BatchNorm3d(channel)
        elif norm_mode == 'gn':
            if num_groups is not None and ch_per_group is not None:
                raise ValueError('Can only choose one, num_groups or ch_per_group')
            if num_groups is not None:
                assert channel%num_groups == 0, 'channel%%num_groups != 0'
                self.norm = nn.GroupNorm(num_groups, channel)
            elif ch_per_group is not None:
                assert channel%ch_per_group == 0, 'channel%%ch_per_group != 0'
                self.norm = nn.GroupNorm(channel//ch_per_group, channel)
            else:
                raise ValueError('Please choose one, num_groups or ch_per_group')
        else:
            raise ValueError('Unknown normalization mode')

    def forward(self, x):
        return self.norm(x)
    
class DeConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale=2):
        super(DeConvBlock, self).__init__()
        k = (scale, scale, 1)
        s = (scale, scale, 1)
        self.deconv = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=k, stride=s, padding=0)
    def forward(self, x):
        x = self.deconv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = (scale, scale, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return x
    
class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale=2, deconv=False):
        super(UpConvBlock, self).__init__()
        if deconv:
            self.up = nn.Sequential(
                DeConvBlock(in_channel, out_channel, scale))
        else:
            layers = [Upsample(scale)]
            if in_channel != out_channel:
                layers.append(nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0))
            self.up = nn.Sequential(*layers)

    def forward(self, x):
        x = self.up(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, n=2):
        super(ConvBlock, self).__init__()
        k = (3, 3, 3)
        p = (1, 1, 1)
        layers = []
        for _ in range(n):
            layers += [
                nn.Conv3d(in_channel, out_channel, kernel_size=k, stride=1, padding=p, bias=bias),
                Norm(out_channel),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x