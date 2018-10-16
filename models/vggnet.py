"""Implementation of VGGNet for experiments """ 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np



# code inspired from torchvision's code style 

class VGGNet(nn.Module):
    def __init__(self, features, ncls):
        super(VGGNet, self).__init__()
        self._feats = features # features are conv layers made by the cfg 
        self.cls = ncls
        self.fc1 = nn.Linear(512*4*4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self._init_layers()
    
    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self._feats(x)
        print(out.size())
        out = out.view(-1, 512*4*4)
        out = F.dropout(F.relu(self.fc1(out)), p=0.5)
        out = F.dropout(F.relu(self.fc2(out)), p=0.5)
        out = self.fc3(out)
        return out 



def make_feats(cfg, use_bnorm=False):
    layers = [] 
    init_ch = 3 
    for p in cfg:
        if p == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(init_ch, p, kernel_size=3, padding=1)
            if use_bnorm:
                layers += [conv2d, nn.BatchNorm2d(p),  nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
            init_ch = p
    return nn.Sequential(*layers)




cfg = {
    'A':[64, 'M',128,'M', 256, 256, 'M', 512, 512]
}


def vgg11(params):
    config = cfg[params.cfg_key]
    feats = make_feats(config)
    model = VGGNet(feats, params.num_classes)
    return model 

def vgg11_bn(params):
    config = cfg[params.cfg_key]
    feats = make_feats(config, use_bnorm=True)
    model = VGGNet(feats, params.num_classes)
    return model 




def loss_fn(out_batch, label_batch):
    return nn.CrossEntropyLoss()(out_batch, label_batch)


def accuracy(outputs, labels):
    """
    Calculate the accuracy of the outputs for a given output, label pair. 

    Outputs and labels must be numpy ndarrays. 
    """
    corr_output = np.argmax(outputs, axis=1)
    acc = np.sum(corr_output==labels)/float(labels.size)
    return acc


metrics = {

    "accuracy":accuracy,
}


    
