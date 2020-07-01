#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/25 15:23
# @Author : caius
# @Site : 
# @File : model.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
import os

sys.path.append('../')
# sys.path.append('../base')
from  base import BaseModel
# from base.base_model import BaseModel
# print(os.path.isdir('../'))



class ResNetBlock(nn.Module):
    def __init__(self, in_depth, depth, first=False):
        super(ResNetBlock, self).__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(depth)
        self.lrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=1, padding=1)
        if not self.first :
            self.pre_bn = nn.BatchNorm2d(in_depth)

    def forward(self, x):
        # x is (B x d_in x T)
        prev = x
        prev_mp =  self.conv11(x)
        if not self.first:
            out = self.pre_bn(x)
            out = self.lrelu(out)
        else:
            out = x
        out = self.conv1(x)
        # out is (B x depth x T/2)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        # out is (B x depth x T/2)
        out = out + prev_mp
        return out


class ConvModel(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(32, 32,  True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = ResNetBlock(32, 32,  False)
        self.block3 = ResNetBlock(32, 32,  False)
        self.block4= ResNetBlock(32, 32, False)
        self.block5= ResNetBlock(32, 32, False)
        self.block6 = ResNetBlock(32, 32, False)
        self.block7 = ResNetBlock(32, 32, False)
        self.block8 = ResNetBlock(32, 32, False)
        self.block9 = ResNetBlock(32, 32, False)
        self.block10=  ResNetBlock(32, 32, False)
        self.block11 = ResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        #out = self.block2(out)
        out = self.mp(out)
        out = self.block3(out)
        #out = self.block4(out)
        out = self.mp(out)
        out = self.block5(out)
        #out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        #out = self.block8(out)
        out = self.mp(out)
        out = self.block9(out)
        #out = self.block10(out)
        out = self.mp(out)
        out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out





if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.zeros(32, 320, 256).to(device)

    model = ConvModel(num_classes=2)
    import time
    print(model)

    tic = time.time()
    y = model(x)
    print(time.time() - tic)
    print(y[0].shape)
    # print(model)
    # inputs = torch.random(3,320,256)
    # inputs.to(device)

    # torch.save(model.state_dict(), 'PAN.pth')
    summary(model, input_size=(32, 320, 256))
