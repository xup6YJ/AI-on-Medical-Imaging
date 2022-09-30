import torch.nn as nn
import torch.nn.functional as F
import math
import torch

# adapt from https://github.com/Kamnitsask/deepmedic

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv3d(inplanes, planes, 3, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        x = x[:, :, 2:-2, 2:-2, 2:-2]
        y[:, :self.inplanes] += x
        y = self.relu(y)
        return y

def conv3x3(inplanes, planes, ksize=3):
    return nn.Sequential(
            nn.Conv3d(inplanes, planes, ksize, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True))

def conv5x5x5(inplanes, planes, ksize=5):
    return nn.Sequential(
            nn.Conv3d(inplanes, planes, ksize, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True))

def conv1x1x1(inplanes, planes, ksize=1):
    return nn.Sequential(
            nn.Conv3d(inplanes, planes, ksize, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True))

def repeat(x, n=3):
    # nc333
    b, c, h, w, t = x.shape
    x = x.unsqueeze(5).unsqueeze(4).unsqueeze(3)
    x = x.repeat(1, 1, 1, n, 1, n, 1, n)
    return x.view(b, c, n*h, n*w, n*t)


class DeepMedic(nn.Module):
    def __init__(self, c=4, n1=30, n2=40, n3=50, m=150, up=True):
        super(DeepMedic, self).__init__()
        #n1, n2, n3 = 30, 40, 50

        n = 2*n3
        self.branch1 = nn.Sequential(
                conv3x3(c, n1),
                conv3x3(n1, n1),
                ResBlock(n1, n2),
                ResBlock(n2, n2),
                ResBlock(n2, n3))

        self.branch2 = nn.Sequential(
                conv3x3(c, n1),
                conv3x3(n1, n1),
                conv3x3(n1, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n3),
                conv3x3(n3, n3))

        self.up3 = nn.Upsample(scale_factor=3,
                mode='trilinear', align_corners=False) if up else repeat

        self.fc = nn.Sequential(
                conv3x3(n, m, 1),
                conv3x3(m, m, 1),
                nn.Conv3d(m, 5, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        x1, x2 = inputs
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x2 = self.up3(x2)
        x = torch.cat([x1, x2], 1)
        x = self.fc(x)
        return x

class VDeepMedic(nn.Module):
    def __init__(self, c=4, n1=30, n2=40, n3=50, m=150, up=True):
        super(VDeepMedic, self).__init__()
        #n1, n2, n3 = 30, 40, 50
        # need 29 inputs

        n = 2*n3
        self.branch1 = nn.Sequential(
                conv3x3(c, n1),
                conv3x3(n1, n1),
                ResBlock(n1, n2),
                ResBlock(n2, n2),
                ResBlock(n2, n2),
                ResBlock(n2, n2),
                ResBlock(n2, n3))

        self.branch2 = nn.Sequential(
                conv3x3(c, n1),
                conv3x3(n1, n1),
                conv3x3(n1, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n3),
                conv3x3(n3, n3))

        self.up3 = nn.Upsample(scale_factor=3,
                mode='trilinear', align_corners=False) if up else repeat

        self.fc = nn.Sequential(
                conv3x3(n, m, 1),
                conv3x3(m, m, 1),
                nn.Conv3d(m, 5, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        x1, x2 = inputs
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x2 = self.up3(x2)
        x = torch.cat([x1, x2], 1)
        x = self.fc(x)
        return x


# no residual connections
class DeepMedicNR(nn.Module):
    def __init__(self, c=1, n1=30, n2=40, n3=50, m=150, up=True):
        super(DeepMedicNR, self).__init__()
        #n1, n2, n3 = 30, 40, 50

        n = 2*n3
        self.branch1 = nn.Sequential(
                conv3x3(c, n1),     #4, 30
                conv3x3(n1, n1),    #30, 30
                conv3x3(n1, n2),    #30, 40
                conv3x3(n2, n2),    #40, 40
                conv3x3(n2, n2),    #40, 40
                conv3x3(n2, n2),    #40, 40
                conv3x3(n2, n3),    #40, 50
                conv3x3(n3, n3))    #50, 50

        self.branch2 = nn.Sequential(
                conv3x3(c, n1),
                conv3x3(n1, n1),
                conv3x3(n1, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n3),
                conv3x3(n3, n3))

        self.up3 = nn.Upsample(scale_factor=3,
                mode='trilinear', align_corners=False) if up else repeat

        self.fc = nn.Sequential(
                conv1x1x1(n, m, 1),  #50, 150
                conv1x1x1(m, m, 1),  #150, 150
                conv1x1x1(m, 2, 1))  #150, 2

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        x1 = inputs
        x2 = x1[:, :, 3:-3, 3:-3, 3:-3]  #crop middle
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x2 = self.up3(x2)
        x = torch.cat([x1, x2], 1)
        x = self.fc(x)
        return x

# import torch
# x1 = torch.rand(100, 1, 25, 25, 25)
# x2 = torch.rand(100, 4, 19, 19, 19)

# x1, x2 = x1.cuda(), x2.cuda()
# model = DeepMedicNR().cuda()
# x = model(x1)
# print(x.size())


# no residual connections


class Conv3x3x3(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0):
    super(Conv3x3x3, self).__init__()
    self.conv3D = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

  def forward(self, x):
    x = F.relu(self.conv3D(x))
    x = self.batch_norm(x)

    return(x)

def conv3x3(inplanes, planes, ksize=3):
    return nn.Sequential(
            nn.Conv3d(inplanes, planes, ksize, bias=False, padding=1),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True))


class DeepMedicNR_test(nn.Module):
    def __init__(self, c=1, n1=30, n2=40, n3=50, m=150, up=True):
        super(DeepMedicNR_test, self).__init__()

        #n1, n2, n3 = 30, 40, 50
        n = 2*n3
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=c, out_channels=n1, kernel_size=3, padding=1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=n1, out_channels=n1, kernel_size=3, padding=1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=n1, out_channels=n2, kernel_size=3, padding=1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=n2, out_channels=n2, kernel_size=3, padding=1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=n2, out_channels=n2, kernel_size=3, padding=1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace=True))
        
        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=n2, out_channels=n2, kernel_size=3, padding=1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential(
            nn.Conv3d(in_channels=n2, out_channels=n3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(
            nn.Conv3d(in_channels=n3, out_channels=n3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace=True))
        
        # self.branch1 = nn.Sequential(
        #         conv3x3(c, n1),     #4, 30
        #         conv3x3(n1, n1),    #30, 30
        #         conv3x3(n1, n2),    #30, 40
        #         conv3x3(n2, n2),    #40, 40
        #         conv3x3(n2, n2),    #40, 40
        #         conv3x3(n2, n2),    #40, 40
        #         conv3x3(n2, n3),    #40, 50
        #         conv3x3(n3, n3))    #50, 50

        self.branch2 = nn.Sequential(
                conv3x3(c, n1),
                conv3x3(n1, n1),
                conv3x3(n1, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n3),
                conv3x3(n3, n3))

        self.up3 = nn.Upsample(scale_factor=2,
                mode='trilinear', align_corners=False) if up else repeat

        self.fc = nn.Sequential(
                conv1x1x1(n, m, 1),  #100, 150
                conv1x1x1(m, m, 1),  #150, 150
                conv1x1x1(m, 1, 1))  #150, 2
        
  
        # self.activation = nn.Sigmoid()

        self.activation = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        x1 = inputs
        # print('input shape: ', x1.shape)
        x2 = x1[:, :, 32:-32, 32:-32, 2:-2]  #crop middle
        # print('x2 input shape: ', x2.shape)
        # x1 = self.branch1(x1)

        x1 = self.conv1(x1)
        # print('output shape conv1: ', x1.shape)
        x1 = self.conv2(x1)
        # print('output shape conv2: ', x1.shape)
        x1 = self.conv3(x1)
        # print('output shape conv3: ', x1.shape)
        x1 = self.conv4(x1)
        # print('output shape conv4: ', x1.shape)
        x1 = self.conv5(x1)
        # print('output shape conv5: ', x1.shape)
        x1 = self.conv6(x1)
        # print('output shape conv6: ', x1.shape)
        x1 = self.conv7(x1)
        # print('output shape conv7: ', x1.shape)
        x1_o = self.conv8(x1)
        # print('output shape conv8: ', x1.shape)
        
        x2_o = self.branch2(x2)
        # print('output shape branch2: ', x2.shape)
        x2_o = self.up3(x2_o)
        out = torch.cat([x1_o, x2_o], 1)
        out = self.fc(out)
        out = self.activation(out)
        return out



# x1 = torch.rand(100, 1, 128, 128, 25)
# # x2 = torch.rand(100, 1, 19, 19, 19)

# # x1 = x1.cuda()
# model = DeepMedicNR()
# # model = model.cuda()
# x = model(x1)
# print(x.size())