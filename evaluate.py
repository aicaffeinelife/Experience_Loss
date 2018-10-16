import argparse 
import logging
import os 
import sys 

import numpy as np 
import torch 
import utils
import torch.nn as nn 
from torch.autograd import Variable 
from models import net 
from tqdm import tqdm 


""" A script to evaluate models """

def evaluate(net, dataloader, loss_fn, params, metrics):

    net.eval()

    summaries = []
    loss_avg = utils.AverageMeter()
    with tqdm(total=len(dataloader)) as t:
        for i, (data, label) in enumerate(dataloader):
            if params.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            # print(data.size())
            # print(label.size())

            # run the input through the net 
            out = net(data)
            # print(out.size())
            loss = loss_fn(out, label)
            loss_avg.update(loss.data[0].item())

            out_batch = out.data.cpu().numpy()
            label_batch = label.data.cpu().numpy()

            summary_batch = {metric:metrics[metric](out_batch, label_batch) for metric in metrics}
            summary_batch['loss'] = loss.data[0].cpu().item()
            summaries.append(summary_batch)
    
    mean_metrics = {metric:np.mean([m[metric] for m in summaries]) for metric in summaries[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in mean_metrics.items())
    logging.info("Val Metrics: "+metrics_string)
    return mean_metrics 


def evaluate_experience(net, teacher_nets, dataloader, loss_fn_exp, params, metrics):
    net.eval()
    for teacher in teacher_nets:
        teacher.eval()

    summaries = [] 
    loss_avg = utils.AverageMeter()
    with tqdm(total=len(dataloader)) as t:
        for i, (data, label) in enumerate(dataloader):
            if params.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            # print(data.size())
            # print(label.size())

            # run the input through the net 
            out = net(data)
            # teach_out = [teacher(data).detach_() for teacher in teacher_nets]
            teach_out = []
            for teacher in teacher_nets:
                out_teacher = teacher(data)
                out_teacher = out_teacher.detach()
                teach_out.append(out_teacher)
            # teach_out.detach_()
            # print(out.size())
            loss = loss_fn_exp(out, teach_out, label, params)
            loss_avg.update(loss.data[0].item())

            out_batch = out.data.cpu().numpy()
            label_batch = label.data.cpu().numpy()

            summary_batch = {metric:metrics[metric](out_batch, label_batch) for metric in metrics}
            summary_batch['loss'] = loss.data[0].cpu().item()
            summaries.append(summary_batch)
    
    mean_metrics = {metric:np.mean([m[metric] for m in summaries]) for metric in summaries[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in mean_metrics.items())
    logging.info("Val Metrics: "+metrics_string)
    return mean_metrics 





