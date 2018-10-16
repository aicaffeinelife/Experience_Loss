""" ResNet model for classification. This model shall later be used as a Teacher """

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1 
    def __init__(self, in_channels, channels, stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        self._ksize = 3
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=self._ksize, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=self._ksize, stride=1, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
    

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        # print("1:{}".format(out.size()))
        out = self.bn2(self.conv2(out))
        # print("2:{}".format(out.size()))

        if self.downsample is not None:
            residual = self.downsample(x) # the downsample is the residual block
        # print("Residual size: {}".format(residual.size()))
        out += residual 
        return F.relu(out)



class BottleNeck(nn.Module):
    expansion = 4 
    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels*self.expansion)
        self.downsample = downsample 
    

    def forward(self, x):
        residual = x 

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(x))

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual 
        return F.relu(out)



class ResNet(nn.Module):
    """
    The base class for any version of ResNet i.e. ResNet 18, ResNet 51, ResNet 101
    args:
    block: The basic building block 
    num_layers: The number of layers of each residual block (given as a list)
    num_classes: The number of classes to predict 
    """
    def __init__(self, block, num_layers, num_classes=10):
        super(ResNet, self).__init__()
        self._in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_layers[0])
        self.layer2 = self._make_layer(block, 128, num_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

    def _make_layer(self, block, in_channels, layer, stride=1):
        downsample = None
        if stride != 1 or self._in_planes != in_channels*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_planes, in_channels*block.expansion, kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(in_channels*block.expansion)
            )
        
        layers = [] 
        layers.append(block(self._in_planes, in_channels, stride, downsample))
        self._in_planes = in_channels*block.expansion 
        for i in range(1, layer):
            layers.append(block(self._in_planes, in_channels))
        
        return nn.Sequential(*layers)
    


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(BottleNeck, [3,4,6,3])



def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)



def accuracy(output, labels):
    corr_output = np.argmax(output, axis=1)
    acc = np.sum(corr_output==labels)/float(labels.size)
    return acc 


metrics = {
    "accuracy":accuracy
}
    



        


