import torch.nn as nn
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from .quaternion_layers import QuaternionConv as QuaternionConv2d
from PIL import Image
#cuda_name='cuda:2'
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU',is_quaternion=False):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation,is_quaternion))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation,is_quaternion))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU',is_quaternion=False):
        super(ConvBatchNorm, self).__init__()
        self.convT = QuaternionConv2d(in_channels, out_channels,kernel_size=1, padding=0)
        self.convT1= nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1)
        self.isQuan=is_quaternion
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):

        if self.isQuan == True:
            out = self.convT(x)
        else:
            out = self.convT1(x)
        out = self.norm(out)
        return self.activation(out)


class ConvBatchNorm_original(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm_original, self).__init__()
        self.convT = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)

        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):

        out = self.convT(x)
        out = self.norm(out)
        return self.activation(out)
class ConvBatchNorm_quaternion(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm_quaternion, self).__init__()
        self.convT = QuaternionConv2d(in_channels, out_channels,kernel_size=1, padding=0)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):

        out = self.convT(x)
        out = self.norm(out)
        return self.activation(out)
class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU',is_quaternion=False):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation,is_quaternion)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU',is_quaternion=False):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation,is_quaternion)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

def compute_midian_infeature(x):


    median_feature=torch.quantile(x,0.5,dim=1,keepdim=True)

    return median_feature
class interaction_fusion(nn.Module):
    def __init__(self, dim, reduction=1):
        super(interaction_fusion, self).__init__()
        #self.conv_mix = nn.Conv2d(dim, dim, kernel_size=3,stride=1, padding=1)
        self.norm = nn.BatchNorm2d(dim)
        self.activation = nn.SiLU()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #self.max_pool
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
    # 在通道维度上进行全局平均池化
        x1_split = torch.chunk(x1, 2, dim=1)
        x1_S0 =  x1_split[0].clone()
        x1_S1 =  x1_split[1].clone()

        x2_split = torch.chunk(x2, 2, dim=1)
        x2_S0 = x2_split[0].clone()
        x2_S1 = x2_split[1].clone()

        x1_s0_mean = torch.mean(x1_S0, dim=1, keepdim=True)
        x2_s0_mean = torch.mean(x2_S0, dim=1, keepdim=True)

        x1_s0_max,_ = torch.max(x1_S0, dim=1, keepdim=True)
        x2_s0_max,_ = torch.max(x2_S0, dim=1, keepdim=True)


        x1_s0_median = compute_midian_infeature(x1_S0)
        x2_s0_median = compute_midian_infeature(x2_S0)


       
        x1_weight= self.sigmoid(x1_s0_max+x1_s0_median)
        x2_weight = self.sigmoid(x2_s0_max+x2_s0_median)



        x1_rebuid = torch.cat([x2_weight*x1_S1,x1_S0],dim=1)
        x2_rebuid = torch.cat([x1_weight*x2_S1,x2_S0],dim=1)


        return x1_rebuid,x2_rebuid

class fusion_feature(nn.Module):
    def __init__(self, dim, reduction=1):
        super(fusion_feature, self).__init__()
        #self.conv_mix = nn.Conv2d(dim, dim, kernel_size=3,stride=1, padding=1)
        self.norm = nn.BatchNorm2d(dim)
        self.activation = nn.ReLU()

    def forward(self, x1, x2):
    # 在通道维度上进行全局平均池化

        x3 = torch.cat((x1, x2), dim=1)
        x4 = F.adaptive_avg_pool2d(x3, (1, 1))
        sort_number = x1.size(1)
        # 为x4的通道添加序号
        sorted_x2, indices = torch.sort(x4, dim=1, descending=True)
        top_indices = indices[:, :sort_number]

        # 根据挑选的通道，从x3中选择相应的特征通道
        T3 = x3.gather(1, top_indices.expand(-1, x1.size(1), x1.size(2), x1.size(3)))
        T3=self.norm(T3)
        T3=self.activation(T3)
        return T3

class UCTransNet(nn.Module):
    def __init__(self, config,n_channels=9, n_classes=1,bilinear=True,img_size=512,vis=False):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Question here
        in_channels = 64
        self.inc = ConvBatchNorm_original(3, in_channels)
        self.inc3 = ConvBatchNorm_quaternion(4, in_channels)


        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2,is_quaternion = False)
        self.down11 = DownBlock(in_channels, in_channels*2, nb_Conv=2,is_quaternion = True)
        self.fusion_feature1 = interaction_fusion(in_channels)

        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2,is_quaternion = False)
        self.down22 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2,is_quaternion = True)
        self.fusion_feature2 = interaction_fusion(in_channels * 2)

        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2,is_quaternion = False)
        self.down33 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2,is_quaternion = True)
        self.fusion_feature3 = interaction_fusion(in_channels * 4)

        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2,is_quaternion = False)
        self.fusion_feature4=fusion_feature(in_channels * 8)

        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2,is_quaternion = False)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2,is_quaternion = False)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2,is_quaternion = False)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2,is_quaternion = False)

        self.outc4 = nn.Conv2d(in_channels*4, n_classes, kernel_size=(1,1))
        self.outc3 = nn.Conv2d(in_channels*2, n_classes, kernel_size=(1,1))
        self.outc2 = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        self.outc1 = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # Question here
        x = x.float()
        #gray_tensor = torch.mean(x, dim=1, keepdim=True)


        B,C,H,W=x.size()
        input_data = (B,1,H,W)
        params = nn.Parameter(torch.empty(*input_data)).to(x.device)
        nn.init.xavier_uniform_(params)

        x1 = self.inc(x)
        T= torch.cat([params,x],dim=1)
        x11 = self.inc3(T) #self.inc(Color.to(x.device))
        x1,x11=self.fusion_feature1(x1, x11)

        x2 = self.down1(x1)
        x22 = self.down11(x11)
        x2, x22=self.fusion_feature2(x2, x22)

        x3 = self.down2(x2)
        x33 = self.down22(x22)
        x3, x33=self.fusion_feature3(x3, x33)

        x4 = self.down3(x3)
        x44 = self.down33(x33)
        x4_skip = self.fusion_feature4(x4, x44)

        x5 = self.down4(x4)

        x = self.up4(x5, x4_skip)
        x_1 = x
        x = self.up3(x, x3)
        x_2 = x
        x = self.up2(x, x2)
        x_3= x
        x = self.up1(x, x1)
        shape = x1.size()[2:]

        x_1 = F.interpolate(x_1, size=shape, mode='bilinear')
        x_2 = F.interpolate(x_2, size=shape, mode='bilinear')
        x_3 = F.interpolate(x_3, size=shape, mode='bilinear')

        logits4 = self.last_activation(self.outc1(x))
        logits3 = self.last_activation(self.outc2(x_3))
        logits2 = self.last_activation(self.outc3(x_2))
        logits1 = self.last_activation(self.outc4(x_1))


        return logits4, logits3, logits2, logits1


