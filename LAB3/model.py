

import torch.nn as nn
import torchvision.models
import torch
import torch.nn.functional as F
from torchsummary import summary
from model_blocks import *




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

        self.up5 = UpConvBlock(init_ch*8, init_ch*8, deconv=deconv)

        self.up_conv0 = ConvBlock(init_ch*2, init_ch)
        self.up_conv1 = ConvBlock(init_ch*4, init_ch*2)
        self.up_conv2 = ConvBlock(init_ch*8, init_ch*4)
        self.up_conv3 = ConvBlock(init_ch*16, init_ch*8)

        self.up_conv4 = ConvBlock(init_ch*8, init_ch*8)

        self.o1 = nn.Sequential(
            # DeConvBlock(init_ch*4, init_ch*4, scale =4),
            nn.Upsample(scale_factor=(4, 4, 1), mode='trilinear'),
            nn.Conv3d(in_channels=init_ch*4, out_channels = 1, kernel_size=1, stride=1, padding=0)
        )

        self.o2 = nn.Sequential(
            # DeConvBlock(init_ch*2, init_ch*2, scale =2),
            nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear'),
            nn.Conv3d(in_channels=init_ch*2, out_channels = 1, kernel_size=1, stride=1, padding=0)
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
        x0 = self.conv0(x)                  #32, 128, 128, 8
        x1 = self.conv1(self.maxpool(x0))   #64, 64, 64, 4
        x2 = self.conv2(self.maxpool(x1))   #128, 32, 32, 2
        x3 = self.conv3(self.maxpool(x2))   #256 ,16, 16, 1
        x4 = self.conv4(self.maxpool(x3))   #512 ,8, 8, 1

        d3 = torch.cat((x3, self.up4(x4)), dim=1) #256+256
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

# class Seg_UNet(BaseUNet):
#     def __init__(self, **kwargs):
#         super(Seg_UNet, self).__init__(**kwargs)

#     def forward(self, x):
#         x0 = self.conv0(x)

#         x1 = self.conv1(self.maxpool(x0))
#         x2 = self.conv2(self.fv1(self.maxpool(x1)))
#         x3 = self.conv3(self.fv2(self.maxpool(x2)))
#         x4 = self.conv4(self.fv3(self.maxpool(x3)))

#         x4 = self.paspp(x4)

#         d3 = torch.cat((x3, self.up4(x4)), dim=1)
#         d3 = self.up_conv3(d3)
        
#         d2 = torch.cat((x2, self.up3(d3)), dim=1)
#         d2 = self.up_conv2(d2)
        

#         d1 = torch.cat((x1, self.up2(d2)), dim=1)
#         d1 = self.up_conv1(d1)

#         d0 = torch.cat((x0, self.up1(d1)), dim=1)
#         d0 = self.up_conv0(d0)

#         out = self.activation(self.conv_1x1(d0))
#         return out  

class Seg_UNet(BaseUNet):
    def __init__(self, **kwargs):
        super(Seg_UNet, self).__init__(**kwargs)

    def forward(self, x):
        x0 = self.conv0(x)

        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.fv1(self.maxpool(x1)))
        x3 = self.conv3(self.fv2(self.maxpool(x2)))
        x4 = self.fv3(self.maxpool(x3))

        x4 = self.paspp(x4)

        d3 = torch.cat((x3, self.up5(x4)), dim=1)
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
        super(DSeg_UNet, self).__init__(**kwargs)

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
    
class ResUNet(nn.Module):
    def __init__(self, **kwargs):
        super(ResUNet, self).__init__(**kwargs)
        in_ch = 1
        out_ch = 1
        init_ch = 32
        deconv = True

        self.conv0 = ResConvBlock(in_ch, init_ch)
        self.conv1 = ResConvBlock(init_ch, init_ch*2)
        self.conv2 = ResConvBlock(init_ch*2, init_ch*4)
        self.conv3 = ResConvBlock(init_ch*4, init_ch*8)
        self.conv4 = ResConvBlock(init_ch*8, init_ch*16)

        self.up1 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up2 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up3 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)
        self.up4 = UpConvBlock(init_ch*16, init_ch*8, deconv=deconv)

        self.up_conv0 = ResConvBlock(init_ch*2, init_ch)
        self.up_conv1 = ResConvBlock(init_ch*4, init_ch*2)
        self.up_conv2 = ResConvBlock(init_ch*8, init_ch*4)
        self.up_conv3 = ResConvBlock(init_ch*16, init_ch*8)

        self.conv_1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
            # self.activation = nn.Identity()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))

        d3 = torch.cat((self.up4(x4), x3), dim=1)
        d3 = self.up_conv3(d3)
        d2 = torch.cat((self.up3(d3), x2), dim=1)
        d2 = self.up_conv2(d2)
        d1 = torch.cat((self.up2(d2), x1), dim=1)
        d1 = self.up_conv1(d1)
        d0 = torch.cat((self.up1(d1), x0), dim=1)
        d0 = self.up_conv0(d0)

        out = self.activation(self.conv_1x1(d0))

        return out
    

class FV_ResUNet(nn.Module):
    def __init__(self, **kwargs):
        super(FV_ResUNet, self).__init__(**kwargs)
        in_ch = 1
        out_ch = 1
        init_ch = 32
        deconv = True

        self.conv0 = ResConvBlock(in_ch, init_ch)
        self.conv1 = ResConvBlock(init_ch, init_ch*2)

        self.fv1 = FVBlock(in_channels = init_ch*2)
        self.conv2 = ResConvBlock(init_ch*2, init_ch*4)

        self.fv2 = FVBlock(in_channels = init_ch*4)
        self.conv3 = ResConvBlock(init_ch*4, init_ch*8)

        self.fv3 = FVBlock(in_channels = init_ch*8)
        self.conv4 = ResConvBlock(init_ch*8, init_ch*16)

        self.up1 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up2 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up3 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)
        self.up4 = UpConvBlock(init_ch*16, init_ch*8, deconv=deconv)

        self.up_conv0 = ResConvBlock(init_ch*2, init_ch)
        self.up_conv1 = ResConvBlock(init_ch*4, init_ch*2)
        self.up_conv2 = ResConvBlock(init_ch*8, init_ch*4)
        self.up_conv3 = ResConvBlock(init_ch*16, init_ch*8)

        self.conv_1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
            # self.activation = nn.Identity()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.fv1(self.maxpool(x1)))
        x3 = self.conv3(self.fv2(self.maxpool(x2)))
        x4 = self.conv4(self.fv3(self.maxpool(x3)))

        d3 = torch.cat((self.up4(x4), x3), dim=1)
        d3 = self.up_conv3(d3)
        d2 = torch.cat((self.up3(d3), x2), dim=1)
        d2 = self.up_conv2(d2)
        d1 = torch.cat((self.up2(d2), x1), dim=1)
        d1 = self.up_conv1(d1)
        d0 = torch.cat((self.up1(d1), x0), dim=1)
        d0 = self.up_conv0(d0)

        out = self.activation(self.conv_1x1(d0))

        return out
    

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
