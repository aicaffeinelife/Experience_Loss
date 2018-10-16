""" A script to analyze a trained student model on the val set """ 

import os 
import json 
import logging
import argparse 
import torch 
import numpy as np
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F
from torchnet.meter import ConfusionMeter 
from tqdm import tqdm 
from dataloader_utils import load_dataset
import utils

from models import net 
import matplotlib.pyplot as plt 
import seaborn as sns 

parser = argparse.ArgumentParser()
parser.add_argument('--param_path', default='experiments/base_cnn_cifar10', help='Path to the folder containing params')
parser.add_argument('--model_name', default='best', help='Path to the model checkpoint to load')
parser.add_argument('--plot_path',  default=None, help='Complete path to save the plot')
parser.add_argument('--text_path', default='', help='Path to save a file as a nptxt')



def analyze_model(net, dataloader, params, num_classes=10):
    """
    Analyze the model by running the data in the val set
    """
    net.eval()
    sfmax_scores = [] 
    preds = [] 
    cfmat = ConfusionMeter(num_classes)
    temp = None
    if params.experiment_type == "base":
        temp = params.temperature
    elif params.experiment_type == "experience":
        temp = params.student_temperature
    


    with tqdm(total=len(dataloader)) as t:

        for ix, (data_batch, label_batch) in enumerate(dataloader):
            if params.cuda:
                data_batch, label_batch = data_batch.cuda(async=True), label_batch.cuda(async=True)
        
            data_batch, label_batch = Variable(data_batch), Variable(label_batch)
            print(data_batch.size())
            output_batch = net(data_batch)
            print(output_batch.size())
            cfmat.add(output_batch.data, label_batch.data)
            sfmax = F.softmax(output_batch/temp, dim=1)
            sfmax_batch = sfmax.data.cpu().numpy()
            sfmax_scores.append(sfmax_batch)


            output_pred = output_batch.data.cpu().numpy()
            label_pred = label_batch.data.cpu().numpy() 

            pred_acc = (np.argmax(output_pred, axis=1)==label_pred).astype(float)
            preds.append(pred_acc.reshape(label_pred.size, 1))
    t.update()
    
    sfmax = np.vstack(sfmax_scores)
    pred_corr = np.vstack(preds)
    return sfmax, pred_corr, cfmat.value().astype(int)





if __name__ == '__main__':
    args = parser.parse_args()
    param_file = os.path.join(args.param_path, 'params.json')
    assert(os.path.isfile(param_file)), 'A valid params.json file was not found at:{}'.format(param_file)
    params = utils.ParamParser(param_file)
    params.cuda = torch.cuda.is_available()
    utils.setLogger(os.path.join(args.param_path, 'analysis.log'))
    logging.info("Starting evaluation ....")
    logging.info("loading validation dataset")
    val_dataset = load_dataset('val', params)
    logging.info("dataset loaded")

    checkpoint_path = os.path.join(args.param_path, args.model_name +'.pth.tar')
    assert(os.path.exists(checkpoint_path)), 'No .pth.tar found at:{}'.format(checkpoint_path)
    model = net.CIFARNet(params).cuda() if params.cuda else net.CIFARNet(params)
    metrics = net.metrics 
    utils.load_checkpoint(checkpoint_path, model)
    sfmax, pred_correct, confusion_matrix = analyze_model(model, val_dataset, params)
    # np.savetxt(args.plot_path, pred_correct)

    heatmap = sns.heatmap(confusion_matrix, cmap="YlGnBu")
    # clustermap = sns.clustermap(confusion_matrix)
    if args.plot_path is not None:
        plt.savefig(args.plot_path)
    
    if args.text_path is not None: 
        np.savetxt(args.text_path, confusion_matrix)

    # an_metrics = {'softmax':sfmax, 'correct_preds':pred_correct, 'confusion_mat':confusion_matrix}

    # for k,v in an_metrics.items():
    #     fname = args.dataset + '_'+str(args.temperature) + k + '_results.txt'
    #     save_path = os.path.join(args.param_path, fname)
    #     np.savetxt(save_path, v)


    
        
