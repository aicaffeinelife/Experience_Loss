""" An implementation of wide ResNet """ 

import torch 
import torch.nn as nn 
import torch.nn.functional as  F
import torch.nn.init as init 

import numpy as np


class WideBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, stride=1):
        super(WideBasicBlock, self).__init__()
        self.dropout = dropout 
        self.stride = stride
        # WRN seems to follow the pipeline BN->Conv->BN
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch) 
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=self.stride, padding=1, bias=True)
        
        self.shortcut = None  # Don't know why this is there (residual block??)
        if self.stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=self.stride, padding=1, bias=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    

    def forward(self, x):
        out = F.dropout(self.conv1(F.relu(self.bn1(x))), p=self.dropout)
        out = self.conv2(F.relu(self.bn2(out)))
        if self.shortcut is not None:
            out += self.shortcut(x)
        return out




class WideResNet(nn.Module):
    def __init__(self, params):
        super(WideResNet, self).__init__()
        self._depth = params.depth 
        self._k = params.widen_factor
        self._drpout = params.dropout
        self.ncls = params.num_classes
        self._in_ch = params.initial_channel
        self._in_planes = 16
        assert((self._depth-4)%6 == 0), 'The depth must be 6x+4'
        self._n = int((self._depth-4)/6)
        self.widths = [16] + [int(v*self._k) for v in (16,32,64)]

        self.conv1 = nn.Conv2d(self._in_ch, self.widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(WideBasicBlock, self.widths[1], self._n, self._drpout, stride=1)
        self.layer2 = self._make_layer(WideBasicBlock, self.widths[2], self._n, self._drpout, stride=2)
        self.layer3 = self._make_layer(WideBasicBlock, self.widths[3], self._n, self._drpout, stride=2)
        self.bn1 = nn.BatchNorm2d(self.widths[3])
        self.linear = nn.Linear(self.widths[3], self.ncls)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 0)
                init.constant_(m.bias, 0)

    

    def _make_layer(self, block, width, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = [] 
        for stride in strides:
            layers.append(block(self._in_planes, width, self._drpout, stride=stride))
            self._in_planes = width 
        return nn.Sequential(*layers)
    

    def forward(self, x):
        out = self.conv1(x)
        print(out.size())
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out)) # this is a curious way of calculating output of conv layer. Why is this faster???? 
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        return out


def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)



def accuracy(output, labels):
    corr_output = np.argmax(output, axis=1)
    acc = np.sum(corr_output==labels)/float(labels.size)
    return acc 


metrics = {
    "accuracy":accuracy
}



