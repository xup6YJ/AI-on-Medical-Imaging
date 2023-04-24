

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
        self.conv1 = ConvRelu(in_channels = self.in_channels, out_channels = self.in_channels, kernel_size = 1, padding = 0)
        self.conv2 = ConvRelu(in_channels = int(self.in_channels*3), out_channels = self.in_channels, kernel_size = 3, padding = 1)

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
        print ('PASPP block', 'output shape: ', output_tensor.shape) 

        return(output_tensor) #element-wise multiplication



class encoder(nn.Module):
    # input = 128*128*64
    def __init__(self, in_channels):
        super(encoder, self).__init__()

        # Layer 1 1->64 64->128
        self.l1 = nn.Sequential(
            ConvRelu(in_channels = in_channels, out_channels = 64, kernel_size = 3, padding = 1),
            ConvRelu(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        )

        # Layer 2 128->256
        self.l2 = nn.Sequential(
            FVBlock(in_channels = 128),
            ConvRelu(in_channels = 128, out_channels = 256, kernel_size = 3,  padding = 1)
        )

        # Layer 3 256->512
        self.l3 = nn.Sequential(
            FVBlock(in_channels = 256),
            ConvRelu(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1)
        )

        # Layer 4 512
        self.l4 = nn.Sequential(
            FVBlock(in_channels = 512),
            PASPP_block(in_channels = 512)
        )

    def forward(self, x):
        down_sampling_feature = []
        print('Input shape: ', x.shape)
        x1 =self.l1(x)
        down_sampling_feature.append(x1)
        print('l1 output shape: ', x1.shape)

        x1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)(x1)
        x2 = self.l2(x1)
        down_sampling_feature.append(x2)
        print('l2 output shape: ', x2.shape)

        x2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)(x2)
        x3 = self.l3(x2)
        down_sampling_feature.append(x3)
        print('l3 output shape: ', x3.shape)

        x3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)(x3)
        x4 = self.l4(x3)
        print('l4 output shape: ', x4.shape)

        return x4, down_sampling_feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"


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
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, output_padding = output_padding),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        print('Deconv shape: ', self.deconv(x).shape)
        return(self.deconv(x))
    

    
class decoder(nn.Module):
    def __init__(self, out_channels):
        super(decoder, self).__init__()
        
        # l1 output shape:  torch.Size([1, 128, 16, 128, 128])
        # l2 output shape:  torch.Size([1, 256, 8, 64, 64])
        # l3 output shape:  torch.Size([1, 512, 4, 32, 32])

        # l4 output shape:  torch.Size([1, 512, 2, 16, 16])

        #l1
        self.deconv1 = deconvblock(512, 512)
        self.conv1 = ConvRelu(in_channels = 512*2, out_channels = 256, kernel_size = 3, padding = 1)

        #l2
        self.deconv2 = deconvblock(256, 256) #256 + 512 -> 128
        self.conv2 = ConvRelu(in_channels = 512, out_channels = 128, kernel_size = 3, padding = 1)

        #l3
        self.deconv3 = deconvblock(128, 128) # 128 + 
        self.conv3 = ConvRelu(in_channels = 256, out_channels = 64, kernel_size = 3, padding = 1)

    def forward(self, x, down_sampling_feature):

        x1 = self.deconv1(x)
        print('x1.shape: ', x1.shape)
        x1 = torch.cat((x1, down_sampling_feature[-1]) , dim = 1)
        print('x1 concat shape: ', x1.shape)
        x1 = self.conv1(x1)

        x2 = self.deconv2(x1)
        x2 = torch.cat((x2, down_sampling_feature[-2]) , dim = 1)
        print('x2 concat shape: ', x2.shape)
        x2 = self.conv2(x2)

        x3 = self.deconv3(x2)
        x3 = torch.cat((x3, down_sampling_feature[-3]) , dim = 1)
        x3 = self.conv3(x3)

        return x3


class COVID_seg(nn.Module):
    def __init__(self, in_channels, out_channels, final_activation = 'softmax'):
        super(COVID_seg, self).__init__()
        self.encoder = encoder(in_channels=1)
        self.decoder = decoder(out_channels=out_channels)
        
        if final_activation == 'sigmoid':
            self.f_activation = nn.Sigmoid()
        else:
            self.f_activation = nn.Softmax(dim = 1)

    def forward(self, x):
        x, d_features = self.encoder(x)
        x = self.decoder(x, d_features)
        output = self.f_activation(x)

        print("Final output shape: ", output.shape)

        return output
    
# if __name__ == '__main__':
#     inputs = torch.randn(1, 512, 16, 128, 128)
#     # inputs = torch.randn(1, 1, 16, 128, 128)
#     inputs = inputs.cuda()

#     deconv1 = deconvblock(512, 256)
#     deconv1.cuda()

#     x_test = deconv1(inputs)


if __name__ == '__main__':
    # inputs = torch.randn(1, 512, 16, 128, 128)
    inputs = torch.randn(1, 1, 16, 128, 128)
    inputs = inputs.cuda()

    model = COVID_seg(1, 1)
    model.cuda()

    x_test = model(inputs)