

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



class encoder(nn.Module):
    # input = 128*128*64
    def __init__(self, in_channels):
        super(encoder, self).__init__()

        # Layer 1 1->64 64->128
        self.l1 = nn.Sequential(
            ConvRelu(in_channels = in_channels, out_channels = 64, kernel_size = 3, padding = 1),
            ConvRelu(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        )

        # Layer 2 128->256
        self.l2 = nn.Sequential(
            FVBlock(in_channels = 64),
            ConvRelu(in_channels = 64, out_channels = 128, kernel_size = 3,  padding = 1),
            ConvRelu(in_channels = 128, out_channels = 128, kernel_size = 3,  padding = 1)
        )

        # Layer 3 256->512
        self.l3 = nn.Sequential(
            FVBlock(in_channels = 128),
            ConvRelu(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1),
            ConvRelu(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        )

        # Layer 4 512
        self.l4 = nn.Sequential(
            FVBlock(in_channels = 256),
            # ConvRelu(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1),
            PASPP_block(in_channels = 256)
        )

    def forward(self, x):
        down_sampling_feature = []
        # print('Input shape: ', x.shape)
        x1 =self.l1(x)
        down_sampling_feature.append(x1)
        # print('l1 output shape: ', x1.shape)

        x1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)(x1)
        x2 = self.l2(x1)
        down_sampling_feature.append(x2)
        # print('l2 output shape: ', x2.shape)

        x2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)(x2)
        x3 = self.l3(x2)
        down_sampling_feature.append(x3)
        # print('l3 output shape: ', x3.shape)

        x3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)(x3)
        x4 = self.l4(x3)
        # print('l4 output shape: ', x4.shape)

        return x4, down_sampling_feature


# if __name__ == '__main__':
#     inputs = torch.randn(1, 1, 16, 128, 128)
#     inputs = inputs.cuda()

#     encoder = encoder(1)
#     # print(encoder)
#     encoder.cuda()

#     x_test = encoder(inputs)


#Decoder 
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
    

    
class decoder(nn.Module):
    def __init__(self, out_channels):
        super(decoder, self).__init__()
        
        # l1 output shape:  torch.Size([1, 128, 16, 128, 128])
        # l2 output shape:  torch.Size([1, 256, 8, 64, 64])
        # l3 output shape:  torch.Size([1, 512, 4, 32, 32])

        # l4 output shape:  torch.Size([1, 512, 2, 16, 16])

        #l1
        self.deconv1 = deconvblock(512, 256)
        self.conv1 = ConvRelu(in_channels = 256*2, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv2 = ConvRelu(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)

        #l2
        self.deconv2 = deconvblock(256, 128) #256 + 512 -> 128
        self.conv3 = ConvRelu(in_channels = 128*2, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv4 = ConvRelu(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)

        #l3
        self.deconv3 = deconvblock(128, 64) # 128 + 
        self.conv5 = ConvRelu(in_channels = 64*2, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv6 = ConvRelu(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)

        self.conv7 = ConvRelu(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1)

    def forward(self, x, down_sampling_feature):

        x1 = self.deconv1(x)
        # print('x1.shape: ', x1.shape)
        x1 = torch.cat((x1, down_sampling_feature[-1]) , dim = 1)
        # print('x1 concat shape: ', x1.shape)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)

        x2 = self.deconv2(x1)
        x2 = torch.cat((x2, down_sampling_feature[-2]) , dim = 1)
        # print('x2 concat shape: ', x2.shape)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)

        x3 = self.deconv3(x2)
        x3 = torch.cat((x3, down_sampling_feature[-3]) , dim = 1)
        x3 = self.conv5(x3)
        x3 = self.conv6(x3)

        x4 = self.conv7(x3)

        return x4

class ds_decoder(nn.Module):
    def __init__(self, out_channels):
        super(ds_decoder, self).__init__()
        
        # l1 output shape:  torch.Size([1, 128, 16, 128, 128])
        # l2 output shape:  torch.Size([1, 256, 8, 64, 64])
        # l3 output shape:  torch.Size([1, 512, 4, 32, 32])

        # l4 output shape:  torch.Size([1, 512, 2, 16, 16])

        #l1
        self.deconv1 = deconvblock(512, 256)
        self.conv1 = ConvReluDrop(in_channels = 256*3, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv2 = ConvRelu(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        #l1-2
        self.upsample1 = nn.Upsample(scale_factor=4, mode='trilinear')
        self.conv3 = nn.Conv3d(in_channels=256, out_channels = 1, kernel_size=1, padding=0)

        #l2
        self.deconv2 = deconvblock(256, 128) #256 + 512 -> 128
        self.conv4 = ConvReluDrop(in_channels = 128*3, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv5 = ConvRelu(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        #l2-2
        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.conv6 = nn.Conv3d(in_channels=128, out_channels = 1, kernel_size=1, padding=0)

        #l3
        self.deconv3 = deconvblock(128, 64) # 128 + 
        self.conv7 = ConvReluDrop(in_channels = 64*3, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv8 = ConvRelu(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv9 = nn.Conv3d(in_channels=64, out_channels = 1, kernel_size=1, padding=0)

    def forward(self, x, down_sampling_feature):

        #l1
        x1 = self.deconv1(x)
        # print('x1.shape: ', x1.shape)
        x1 = torch.cat((x1, down_sampling_feature[-1]) , dim = 1)
        # print('x1 concat shape: ', x1.shape)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        #l1-2
        x1_1 = self.upsample1(x1)
        x1_1 = self.conv3(x1_1)

        #l2
        x2 = self.deconv2(x1)
        x2 = torch.cat((x2, down_sampling_feature[-2]) , dim = 1)
        # print('x2 concat shape: ', x2.shape)
        x2 = self.conv4(x2)
        x2 = self.conv5(x2)
        #l2-2
        x2_1 = self.upsample2(x2)
        x2_1 = self.conv6(x2_1)

        #l3
        x3 = self.deconv3(x2)
        x3 = torch.cat((x3, down_sampling_feature[-3]) , dim = 1)
        x3 = self.conv7(x3)
        x3 = self.conv8(x3)

        x4 = self.conv9(x3)

        return x1_1, x2_1, x4


class COVID_seg(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, final_activation = 'sigmoid'):
        super(COVID_seg, self).__init__()
        self.encoder = encoder(in_channels=in_channels)
        self.decoder = decoder(out_channels=out_channels)

        if final_activation == 'sigmoid':
            self.f_activation = nn.Sigmoid()
        else:
            self.f_activation = nn.Softmax(dim = 1)

    def forward(self, x):
        x, d_features = self.encoder(x)
        x = self.decoder(x, d_features)
        output = self.f_activation(x)

        # print("Final output shape: ", output.shape)

        return output
    

class DS_COVID_seg(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, final_activation = 'sigmoid'):
        super(DS_COVID_seg, self).__init__()
        self.encoder = encoder(in_channels=in_channels)
        self.decoder = ds_decoder(out_channels=out_channels)

        if final_activation == 'sigmoid':
            self.f_activation = nn.Sigmoid()
        else:
            self.f_activation = nn.Softmax(dim = 1)

    def forward(self, x):
        x, d_features = self.encoder(x)
        x1, x2, x3 = self.decoder(x, d_features)
        output = self.f_activation(x1)
        output2 = self.f_activation(x2)
        output3 = self.f_activation(x3)

        # print("Final output shape: ", output.shape)

        return output, output2, output3
    
# if __name__ == '__main__':
#     inputs = torch.randn(1, 512, 16, 128, 128)
#     # inputs = torch.randn(1, 1, 16, 128, 128)
#     inputs = inputs.cuda()

#     deconv1 = deconvblock(512, 256)
#     deconv1.cuda()

#     x_test = deconv1(inputs)


'''
UNet
'''
epsilon = 1e-7

norm_mode = 'bn'
num_groups = None
ch_per_group = 16

class Norm(nn.Module):
    def __init__(self, channel):
        super(Norm, self).__init__()
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
    

class BaseUNet(nn.Module):
    def __init__(self, **kwargs):
        super(BaseUNet, self).__init__(**kwargs)
        in_ch = 1
        out_ch = 1
        init_ch = 32
        deconv = True

        self.conv0 = ConvBlock(in_ch, init_ch)
        self.conv1 = ConvBlock(init_ch, init_ch*2)

        self.fv1 = FVBlock(in_channels = init_ch*2)
        self.conv2 = ConvBlock(init_ch*2, init_ch*4)

        self.fv2 = FVBlock(in_channels = init_ch*4)
        self.conv3 = ConvBlock(init_ch*4, init_ch*8)

        self.fv3 = FVBlock(in_channels = init_ch*8)
        self.conv4 = ConvBlock(init_ch*8, init_ch*16)

        # self.fv4 = FVBlock(in_channels = init_ch*16)
        self.paspp = PASPP_block(in_channels = init_ch*16)

        self.up1 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up2 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up3 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)
        self.up4 = UpConvBlock(init_ch*16, init_ch*8, deconv=deconv)

        self.up_conv0 = ConvBlock(init_ch*2, init_ch)
        self.up_conv1 = ConvBlock(init_ch*4, init_ch*2)
        self.up_conv2 = ConvBlock(init_ch*8, init_ch*4)
        self.up_conv3 = ConvBlock(init_ch*16, init_ch*8)

        self.o1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear'),
            nn.Conv3d(in_channels=init_ch*4, out_channels = 1, kernel_size=1, padding=0)
        )

        self.o2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels=init_ch*2, out_channels = 1, kernel_size=1, padding=0)
        )


        self.conv_1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)


class UNet(BaseUNet):
    def __init__(self, **kwargs):
        super(UNet, self).__init__(**kwargs)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))

        d3 = torch.cat((x3, self.up4(x4)), dim=1)
        d3 = self.up_conv3(d3)
        d2 = torch.cat((x2, self.up3(d3)), dim=1)
        d2 = self.up_conv2(d2)
        d1 = torch.cat((x1, self.up2(d2)), dim=1)
        d1 = self.up_conv1(d1)
        d0 = torch.cat((x0, self.up1(d1)), dim=1)
        d0 = self.up_conv0(d0)


        out = self.activation(self.conv_1x1(d0))
        return out
    

class FV_UNet(BaseUNet):
    def __init__(self, **kwargs):
        super(FV_UNet, self).__init__(**kwargs)

    def forward(self, x):
        x0 = self.conv0(x)

        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.fv1(self.maxpool(x1)))
        x3 = self.conv3(self.fv2(self.maxpool(x2)))
        x4 = self.conv4(self.fv3(self.maxpool(x3)))

        d3 = torch.cat((x3, self.up4(x4)), dim=1)
        d3 = self.up_conv3(d3)
        d2 = torch.cat((x2, self.up3(d3)), dim=1)
        d2 = self.up_conv2(d2)
        d1 = torch.cat((x1, self.up2(d2)), dim=1)
        d1 = self.up_conv1(d1)
        d0 = torch.cat((x0, self.up1(d1)), dim=1)
        d0 = self.up_conv0(d0)


        out = self.activation(self.conv_1x1(d0))
        return out

class Seg_UNet(BaseUNet):
    def __init__(self, **kwargs):
        super(FV_UNet, self).__init__(**kwargs)

    def forward(self, x):
        x0 = self.conv0(x)

        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.fv1(self.maxpool(x1)))
        x3 = self.conv3(self.fv2(self.maxpool(x2)))
        x4 = self.conv4(self.fv3(self.maxpool(x3)))

        x4 = self.paspp(x4)

        d3 = torch.cat((x3, self.up4(x4)), dim=1)
        d3 = self.up_conv3(d3)
        
        d2 = torch.cat((x2, self.up3(d3)), dim=1)
        d2 = self.up_conv2(d2)
        

        d1 = torch.cat((x1, self.up2(d2)), dim=1)
        d1 = self.up_conv1(d1)

        d0 = torch.cat((x0, self.up1(d1)), dim=1)
        d0 = self.up_conv0(d0)

        out = self.activation(self.conv_1x1(d0))
        return out  
    
class DSeg_UNet(BaseUNet):
    def __init__(self, **kwargs):
        super(FV_UNet, self).__init__(**kwargs)

    def forward(self, x):
        x0 = self.conv0(x)

        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.fv1(self.maxpool(x1)))
        x3 = self.conv3(self.fv2(self.maxpool(x2)))
        x4 = self.conv4(self.fv3(self.maxpool(x3)))

        x4 = self.paspp(x4)

        d3 = torch.cat((x3, self.up4(x4)), dim=1)
        d3 = self.up_conv3(d3)

        d2 = torch.cat((x2, self.up3(d3)), dim=1)
        d2 = self.up_conv2(d2)
        o1 = self.activation(self.o1(d2))  #Check

        d1 = torch.cat((x1, self.up2(d2)), dim=1)
        d1 = self.up_conv1(d1)
        o2 = self.activation(self.o2(d1))  #Check

        d0 = torch.cat((x0, self.up1(d1)), dim=1)
        d0 = self.up_conv0(d0)

        out = self.activation(self.conv_1x1(d0))
        return o1, o2, out 
    

###Main 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    # inputs = torch.randn(1, 512, 16, 128, 128)
    inputs = torch.randn(1, 1, 8, 128, 128)
    inputs = inputs.cuda()

    model = COVID_seg()
    model = DS_COVID_seg()
    model = UNet()
    model.cuda()

    x_test = model(inputs)

    # from torchsummary import summary

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # "cuda" if torch.cuda.is_available() else "cpu"

    # model = COVID_seg(1, 1)
    # model = model.to(device)
    # summary(model, input_size=(1, 16, 128, 128))
