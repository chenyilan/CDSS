# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       main
   Project Name:    Limited-view detector synthesis
   Author :         Hengrong LAN
   Date:            2019/12/27
   Device:          GTX1080Ti
-------------------------------------------------
   Change Activity:
                   2019/12/26:
-------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from networks.inception_transformer_block import InceptionTransformer




def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)



class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)


        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool



class Bottom(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Bottom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)

        return x

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
            mode=self.up_mode)

        if self.merge_mode == 'add':
            self.conv1 = conv3x3(
                self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same,concat
            self.conv1 = conv3x3(2*self.out_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down1: tensor from the data encoder pathway
            from_down2: tensor from the das encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'add':
            x = from_up + from_down

        else:
            #concat
            x = torch.cat((from_up, from_down), 1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        return x


# Model 1  Unet 
class UNet(nn.Module):


    def __init__(self,  in_channels=3, depths=None, embed_dims=None, num_heads=None,num_patches=None, attention_heads=None, up_mode='transpose', merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.in_channels = in_channels
        st1_idx = 0
        st2_idx = sum(depths[:1])
        st3_idx = sum(depths[:2])
        st4_idx = sum(depths[:3])
        self.down1 = DownConv(self.in_channels,63)
        self.down2 = DownConv(63,192)
        self.down3 = DownConv(192,320)
        self.down4 = DownConv(320,447)

        self.iformer1 = InceptionTransformer(depths=depths[0],
                                            embed_dims=63,
                                            in_chans = 63,
                                            num_heads=num_heads[0],
                                            num_patches=num_patches,
                                            attention_heads=attention_heads[0:2],
                                            num_classes=0,
                                            use_layer_scale=False, layer_scale_init_value=1e-6)
        self.iformer2 = InceptionTransformer(depths=depths[1],
                                            embed_dims=192,
                                            in_chans = 192,
                                            num_heads=num_heads[1],
                                            num_patches=num_patches//2,
                                            attention_heads=attention_heads[2:4],
                                            num_classes=0,
                                            use_layer_scale=False, layer_scale_init_value=1e-6)
        self.iformer3 = InceptionTransformer(depths=depths[2],
                                            embed_dims=320,
                                            in_chans = 320,
                                            num_heads=num_heads[2],
                                            num_patches=num_patches//4,
                                            attention_heads=attention_heads[4:6],
                                            num_classes=0,
                                            use_layer_scale=False, layer_scale_init_value=1e-6)
        self.iformer4 = InceptionTransformer(depths=depths[3],
                                            embed_dims=447,
                                            in_chans = 447,
                                            num_heads=num_heads[3],
                                            num_patches=num_patches//8,
                                            attention_heads=attention_heads[6:8],
                                            num_classes=0,
                                            use_layer_scale=False, layer_scale_init_value=1e-6)
        
        self.bottom = Bottom(447,512)
        self.up1 = UpConv(512,447,merge_mode=self.merge_mode)
        self.up2 = UpConv(447,320,merge_mode=self.merge_mode)
        self.up3 = UpConv(320,192,merge_mode=self.merge_mode)
        self.up4 = UpConv(192,63,merge_mode=self.merge_mode)
        self.outp = conv1x1(63,self.in_channels)
        #self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, img):
        # input: 256,256,60
        bx1,bxbefore_pool1= self.down1(img) # 128,128,64
        bx2,bxbefore_pool2= self.down2(bx1) # 64,64,128
        bx3,bxbefore_pool3= self.down3(bx2) # 32,32,128
        bx4,bxbefore_pool4= self.down4(bx3) # 16,16,256
        bx5 = self.bottom(bx4) # 16, 16, 1024
        bxbefore_pool1=self.iformer1(bxbefore_pool1)
        bxbefore_pool2=self.iformer2(bxbefore_pool2)
        bxbefore_pool3=self.iformer3(bxbefore_pool3)
        bxbefore_pool4=self.iformer4(bxbefore_pool4)

        out = self.up1(bxbefore_pool4, bx5)# 32, 32,512
        out = self.up2(bxbefore_pool3, out)# 64, 64,256
        out = self.up3(bxbefore_pool2, out)# 128, 128,128
        out = self.up4(bxbefore_pool1, out)# 256, 256,64
        out = self.outp(out)
        return out

if __name__ == "__main__":
    """
    testing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device =  torch.device('cuda:1')


    x = Variable(torch.FloatTensor(np.random.random((1, 60, 256, 256))),requires_grad = True).to(device)
    img = Variable(torch.FloatTensor(np.random.random((18, 1, 128, 128))), requires_grad=True).to(device)
    depths = [2, 2, 2, 2]
    #embed_dims = [96, 192, 320, 384]
    num_heads = [3, 6, 10, 12]
    attention_heads = [1] * 2 + [3] * 2 + [7] * 2 + [11] * 2
    img_size = 128
    num_patches = 16
    model = UNet(in_channels=1, depths=depths, embed_dims=None, num_heads=num_heads,num_patches=num_patches, attention_heads=attention_heads,  merge_mode='concat').to(device)
    out = model(img)
    print(out.shape)
    #out = F.upsample(out, (128, 128), mode='bilinear')
    loss = torch.mean(out)

    loss.backward()

    print(loss)
