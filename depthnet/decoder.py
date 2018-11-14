import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init

class deconv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,stride_):
        super(deconv, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch,3,stride=stride_,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #nn.DataParallel(self.deconv)
        )

    def forward(self, x):
        x = self.deconv(x)
        return x

class conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,stride_):
        super(conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride_,padding=1),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(inplace=True),
            #nn.DataParallel(self.conv)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class decoder(nn.Module):

    def __init__(self,hidden_sizes):
      
        super(decoder, self).__init__()
        self.h=hidden_sizes
        self.deconv1=deconv(self.h[4],self.h[4],1)
        self.conv1=conv(self.h[4]*2,self.h[4],1)
        self.deconv2=deconv(self.h[4],self.h[3],1) #2
        self.conv2=conv(self.h[3]*2,self.h[3],1)
        self.deconv3=deconv(self.h[3],self.h[2],1) #2
        self.conv3=conv(self.h[2]*2,self.h[2],1)
        self.deconv4=deconv(self.h[2],self.h[1],1) #2
        self.conv4=conv(self.h[1]*2,self.h[1],1)
        self.deconv5=deconv(self.h[1],self.h[0],1) #2
        #self.conv5=conv(64,32,1)
        self.conv5=nn.Conv2d(self.h[0]*2,self.h[0],3,stride=1,padding=1)
        self.batch5=nn.BatchNorm2d(self.h[0])
        self.sigmoid=nn.Sigmoid()
        self.conv6=nn.Conv2d(self.h[0],1,1,stride=1)
        #self.bn=nn.BatchNorm2d(1)


    def forward(self, x, hiddenList):
        h=hiddenList
        deconv_x=self.deconv1(x)
        concate_x=torch.cat([deconv_x,h[4]], dim=1)
        conv_x=self.conv1(concate_x)

        deconv_x=self.deconv2(conv_x)
        concate_x=torch.cat([deconv_x,h[3]], dim=1)
        conv_x=self.conv2(concate_x)

        deconv_x=self.deconv3(conv_x)
        concate_x=torch.cat([deconv_x,h[2]], dim=1)
        conv_x=self.conv3(concate_x)

        deconv_x=self.deconv4(conv_x)
        concate_x=torch.cat([deconv_x,h[1]], dim=1)
        conv_x=self.conv4(concate_x)

        deconv_x=self.deconv5(conv_x)
        concate_x=torch.cat([deconv_x,h[0]], dim=1)
        #conv_x=self.conv5(concate_x)
        conv_x=self.conv5(concate_x)
        pred_x=self.batch5(conv_x)
        pred_x=self.sigmoid(pred_x)
        pred_x=self.conv6(conv_x)

        return pred_x
