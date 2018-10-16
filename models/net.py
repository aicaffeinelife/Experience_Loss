"""Basic CNN for evaluation with Distillation Loss"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import logging
import numpy as np 

"""
All scripts in the model directory follow the same pattern: 
A model defined as a class, a "loss_function" method to get the loss 
produced by the net and a "accuracy" method to measure accuracy. 

Each model must expose a "metrics" dictionary that contains a mapping from 
a string to a method.
"""
class CIFARNet(nn.Module):
    """
    A basic convnet designed to work on 
    CIFAR dataset. The number
    of channels is provided in the params 
    dictionary which is parsed from a .json 
    file.
    """
    def __init__(self, params):
        super(CIFARNet, self).__init__()
        self._num_channels = params.num_channels
        self._inital_channel = params.initial_channel
        self._ksize = params.kernel_size
        self.conv1 = nn.Conv2d(self._inital_channel, self._num_channels, self._ksize, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self._num_channels)
        self.conv2 = nn.Conv2d(self._num_channels, self._num_channels*2, self._ksize, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self._num_channels*2)
        self.conv3 = nn.Conv2d(self._num_channels*2, self._num_channels*4, self._ksize, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self._num_channels*4)
        self.conv4 = nn.Conv2d(self._num_channels*4, self._num_channels*4, self._ksize, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(self._num_channels*4)
        # the features of the last convnet layer after pooling is self._num_channels*4*4*4 (32->16->8->4)

        self.fc1 = nn.Linear(4*4*self._num_channels*4, self._num_channels*4) 
        self.fc2 = nn.Linear(self._num_channels*4, 10)
        self.dropout_rate = params.dropout_rate 
        # self.mode = mode
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def forward(self, x):
        x = self.bn1(self.conv1(x)) # 32x32
        # print(x.size())
        x = F.relu(F.max_pool2d(x, 2)) # 16x16
        # print(x.size())
        x = self.bn2(self.conv2(x)) #16x16
        # print(x.size())
        x = F.relu(F.max_pool2d(x, 2)) # 8x8
        # print(x.size())
        x = self.bn3(self.conv3(x)) # 8x8
        # print(x.size())
        x = F.relu(F.max_pool2d(x, 2)) # 4x4
        # print(x.size())
        x = F.relu(self.bn4(self.conv4(x)))
        # x = F.max_pool2d(x, 2) 

        x = x.view(-1, 4*4*self._num_channels*4)
        # print(x.size())
        if self.dropout_rate > 0 :
            x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_rate, training=self.training)
        else:
            x = F.relu(self.fc1(x))
        return self.fc2(x)


class CIFARNet2(nn.Module):
    """
    A basic convnet designed to work on 
    CIFAR dataset. The number
    of channels is provided in the params 
    dictionary which is parsed from a .json 
    file. The only difference is that this one batchnormalizes the inputs first
    before passing them through CNNs
    """
    def __init__(self, params):
        super(CIFARNet2, self).__init__()
        self._num_channels = params.num_channels
        self._inital_channel = params.initial_channel
        self._ksize = params.kernel_size
        self.conv1 = nn.Conv2d(self._inital_channel, self._num_channels, self._ksize, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self._num_channels)
        self.conv2 = nn.Conv2d(self._num_channels, self._num_channels*2, self._ksize, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self._num_channels*2)
        self.conv3 = nn.Conv2d(self._num_channels*2, self._num_channels*4, self._ksize, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self._num_channels*4)
        self.conv4 = nn.Conv2d(self._num_channels*4, self._num_channels*4, self._ksize, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(self._num_channels*4)
        # the features of the last convnet layer after pooling is self._num_channels*4*4*4 (32->16->8->4)

        self.fc1 = nn.Linear(4*4*self._num_channels*4, self._num_channels*4) 
        self.fc2 = nn.Linear(self._num_channels*4, 10)
        self.dropout_rate = params.dropout_rate 
        # self.mode = mode
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def forward(self, x):
        x = self.conv1(self.bn1(x)) # 32x32
        # print(x.size())
        x = F.relu(F.max_pool2d(x, 2))# 16x16
        # print(x.size())
        x = self.conv2(F.relu(self.bn2(x))) #16x16
        # print(x.size())
        x = F.relu(F.max_pool2d(x, 2)) # 8x8
        # print(x.size())
        x = F.relu(self.conv3(self.bn3(x))) # 8x8
        # print(x.size())
        x = F.max_pool2d(x, 2) # 4x4
        # print(x.size())
        # x = F.relu(self.bn4(self.conv4(x)))
        # x = F.max_pool2d(x, 2) 

        x = x.view(-1, 4*4*self._num_channels*4)
        # print(x.size())
        if self.dropout_rate > 0 :
            x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_rate, training=self.training)
        else:
            x = F.relu(self.fc1(x))
        return self.fc2(x)








class MNISTNet(nn.Module):
    def __init__(self, params):
        super(MNISTNet, self).__init__()
        self._num_ch = params.num_channels 
        self._ksize = params.kernel_size 
        self._init_ch = params.initial_channel 

        self.conv1 = nn.Conv2d(self._init_ch, self._num_ch, self._ksize, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self._num_ch)
        self.conv2 = nn.Conv2d(self._num_ch, self._num_ch*2, self._ksize, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self._num_ch*2)

        self.fc1 = nn.Linear(7*7*self._num_ch*2, self._num_ch*2)
        self.bfc1 = nn.BatchNorm1d(self._num_ch*2)
        self.fc2 = nn.Linear(self._num_ch*2, 10)
        self.dropout_rate = params.dropout_rate 
    
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(F.max_pool2d(x,2))
        x = self.bn2(self.conv2(x))
        x = F.relu(F.max_pool2d(x,2))
        x = x.view(-1, 7*7*self._num_ch*2)
        if self.dropout_rate > 0 :
            x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_rate, training=self.training)
        else:
            x = F.relu(self.fc1(x))
        return self.fc2(x)


def loss_function(outputs, labels):
    """
    Calculates the loss between the outputs and the labels. 
    outputs: (torch.Variable) or (torch.cuda.Variable)
    labels: (torch.Variable) or (torch.cuda.Variable)

    """
    return nn.CrossEntropyLoss()(outputs, labels)


def loss_function_distillation(outputs, teacher_outputs, labels, params):
    """
    Implements the distillation loss in Hinton et al's paper.
    Args:
    teacher_outputs: Variable containing the outputs of the teacher model 
    outputs: Variable containing the ouput of the current model. 
    labels: Labels 
    params: A dictionary provided by parsing params.json 
    """
    alpha = params.alpha 
    T = params.temperature
    div_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1))
    cross_ent = F.cross_entropy(outputs, labels)
    KDLoss = (T*T*alpha)*div_loss + (1.- alpha)*cross_ent 

    return KDLoss


def aggregate_losses(teach_ops, ops, params):
    """
    Aggregates KL loss with multiple 
    teachers. Since pyTorch doesn't 
    have a reduce_sum function we're 
    gonna write our own
    teach_ops: List of op variables from different teacher models 
    ops: ops of the student model 
    params: The params
    """
    losses = [] 
    T  = params.temperature

    for to in teach_ops:
        div_loss = 0
        div_loss = nn.KLDivLoss(size_average=False)(F.log_softmax(ops/T, dim=1), F.softmax(to/T, dim=1))
        losses.append(div_loss/params.batch_size)
    teach_loss = torch.sum(torch.Tensor(losses).cuda(), dim=0)
    return teach_loss


def aggregate_losses_diff(teach_ops, ops, params):
    """
    Aggregate losses if the temperatures of teachers are 
    different. 
    """
    losses = [] 
    T_lst = params.temperature 
    T_m = max(T_lst)
    assert(len(T_lst) == len(teach_ops))
    div_loss = 0
    for i in range(len(teach_ops)):
        div_loss += nn.KLDivLoss()(F.log_softmax(ops/T_m, dim=1), F.softmax(teach_ops[i]/T_lst[i], dim=1)).detach_()
        # losses.append(div_loss.detach())
    
    # teach_loss = torch.Tensor(losses).cuda() if params.cuda else torch.Tensor(losses)
    # teach_loss = torch.sum(teach_loss, dim=0)
    
    # teach_loss = torch.sum(torch.Tensor(losses).cuda() if params.cuda else torch.Tensor(losses), dim=0)
    return div_loss


def agg_losses_diff_temp(teach_ops, ops, params):
    """
    Aggregate losses for T = [T1, T2] and student 
    temperature Tz 
    """
    T_lst = params.temperature 
    T_z = params.student_temperature 
    teach_losses = []
    for i, teacher in enumerate(teach_ops):
        l = 0 
        print("i:{}, temp:{}".format(i, T_lst[i]))
        l = nn.KLDivLoss()(F.log_softmax(ops/T_z, dim=1), F.softmax(teacher/T_lst[i], dim=1))
        # l = l/params.batch_size
        print("Loss found:{}".format(l))
        teach_losses.append(l)
    
    teach_losses = torch.FloatTensor(teach_losses).cuda() if params.cuda else torch.FloatTensor(teach_losses)
    agg_loss  = torch.sum(teach_losses)
    return agg_loss 



def loss_function_experience(outputs, teacher_outputs, labels, params):
    """
    Implements Experience loss that tries to distill the experience 
    from multiple teachers.
    Args:
    teacher_outputs: List of outputs from teachers queried at runtime
    outputs: The actual outputs produced by the current model 
    labels: Labels 
    params: The parameters for the network
    """
    alpha = params.alpha 
    T_z = params.student_temperature
    # T_m = max(T_lst)
    # agg_loss = 0 
    # for i, ops in enumerate(teacher_outputs):
    #     loss = 0
    #     loss = nn.KLDivLoss()(F.log_softmax(outputs/T_m, dim=1), F.softmax(ops/T_lst[i], dim=1))
    #     agg_loss += loss.detach() 
    agg_loss = agg_losses_diff_temp(teacher_outputs, outputs, params)    

    # agg_loss = aggregate_losses_diff(teacher_outputs, outputs, params)
    # logging.info("Loss from teachers:{}".format(agg_loss))
    ce_loss = F.cross_entropy(outputs, labels)

    ELoss = (alpha*T_z)*agg_loss + (1. - alpha)*ce_loss
    return ELoss





# def loss_function_experience_2(outputs, teacher_outputs, labels,params):
#     """
#     A different take on the experience loss function. Instead of 
#     summing the KL divergence loss, this takes the KL divergence of the 
#     sum of the temperature bound teacher logits and the student logits. 
#     in the same weighted average. 
#     """
#     alpha = params.alpha 
#     T_z = params.student_temperature
#     T_lst = params.temperature



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


