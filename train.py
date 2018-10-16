import argparse 
import os 
import numpy as np 
import logging 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable 
import torch.optim as optim 
import json 
import models.net as net 
import models.resnet as resnet
import models.wide_resnet as wresnet
import models.vggnet as vggnet
import utils
from tqdm import tqdm
from evaluate import evaluate, evaluate_experience
from dataloader_utils import load_dataset
# from tensorboardX import SummaryWriter
from reporter import Reporter
from plotter import Plotter
from csv_reporter import CSVReporter



"""
A script to train models using checkpointing and the evaluation at fixed steps on val set. 
The machinery has been inspired from cs230 @ Stanford University. 
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--param_path', default=None, help="Path to the folder having params.json")
parser.add_argument('--resume_path', default=None, help='Path to any previous saved checkpoint')


def train(net, dataloader, loss_fn, params, metrics, optimizer):
    """
    Train the net for one epoch i.e 1..len(dataloader)
    net: The model to test
    params: The hyperparams 
    loss_fn: The loss function
    metrics: The metrics dictionary containing evaluation metrics. 
    """
    net.train()

    summaries = [] 
    loss_avg = utils.AverageMeter()
    with tqdm(total=len(dataloader)) as t:
        for i, (data_batch, label_batch) in enumerate(dataloader):
            if params.cuda:
                data_batch, label_batch = data_batch.cuda(), label_batch.cuda()
            
            data_batch, label_batch = Variable(data_batch), Variable(label_batch)

            # print(data_batch.size())
            # print(label_batch.size())

            output_batch = net(data_batch)
            loss = loss_fn(output_batch, label_batch)
            print("Output size:{}".format(output_batch.size()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % params.save_summary_steps == 0 :
                print("Saving summaries....")
                out_np = output_batch.data.cpu().numpy()
                label_np = label_batch.data.cpu().numpy()
                batch_summary = {metric: metrics[metric](out_np, label_np) for metric in metrics}
                batch_summary['loss'] = loss.data[0].cpu().item()
                summaries.append(batch_summary)
            
            loss_avg.update(loss.data[0].cpu().item())
            # writer.add_scalar('train_loss/iter', loss.data[0], i)
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    
    # compute mean of all the metrics

    mean_metrics = {metric:np.mean([m[metric] for m in summaries]) for metric in summaries[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in mean_metrics.items())
    logging.info("Train Metrics: "+ metrics_string)
    return mean_metrics


def train_and_eval(net, train_loader, val_loader, optimizer, loss_fn, metrics, params, model_dir, reporter, restore=None):
    """
    Train and evaluate every epoch of a model.
    net: The model. 
    train/val loader: The data loaders
    params: The parameters parsed from JSON file 
    restore: if there is a checkpoint restore from that point. 
    """
    best_val_acc = 0.0 
    if restore is not None:
        restore_file = os.path.join(args.param_path, args.resume_path + '_pth.tar')
        if not os.path.isfile(restore_file):
            logging.info("First epoch, checkpoint not formed yet!")
        else:
            logging.info("Loaded checkpoints from:{}".format(restore_file))
            utils.load_checkpoint(restore_file, net, optimizer)

      
    for ep in range(params.num_epochs):
        logging.info("Running epoch: {}/{}".format(ep+1, params.num_epochs))
        
        # train one epoch 
        train_metrics = train(net, train_loader, loss_fn, params, metrics, optimizer)

        val_metrics = evaluate(net, val_loader, loss_fn, params, metrics)
        reporter.report(ep, 'train_loss', train_metrics['loss'])
        reporter.report(ep, 'train_acc', train_metrics['accuracy'])
        reporter.report(ep, 'val_loss', val_metrics['loss'])
        reporter.report(ep, 'val_acc', val_metrics['accuracy'])
        val_acc = val_metrics['accuracy']
        # writer.add_scalar('val_accuracy/epoch', val_acc, ep)
        # writer.add_scalar('val_loss/epoch', val_metrics['loss'], ep)
        isbest = val_acc >= best_val_acc 

        utils.save_checkpoint({"epoch":ep, "state_dict":net.state_dict(), "optimizer":optimizer.state_dict()}, 
        isBest=isbest, ckpt_dir=model_dir)
    
        if isbest:
            # if the accuracy is great  save it to best.json 
            logging.info("New best accuracy found!")
            best_val_acc = val_acc 
            best_json_path = os.path.join(model_dir, "best_model_params.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
        
        last_acc_path = os.path.join(model_dir, 'last_acc_metrics.json')
        utils.save_dict_to_json(val_metrics, last_acc_path)



def train_experience(net, teacher_nets, dataloader, loss_fn_exp, params, metrics, optimizer):
    """
    Train a student model using experience loss for one epoch:
    net: the student net
    teacher_nets: A list of restored teacher_nets 
    dataloader: The train data loader 
    loss_fn: The experience_loss function 
    params: The params dictionary 
    metrics: Accuracy 
    optimizer: The optimizer 
    """
    net.train() 
    for teacher in teacher_nets:
        teacher.eval()
    # [teacher.eval() for teacher in teacher_nets]
    summaries = []
    loss_avg = utils.AverageMeter()
    
    with tqdm(total=len(dataloader)) as t: 
        for i, (data_batch, label_batch) in enumerate(dataloader):
            if params.cuda:
                data_batch, label_batch = data_batch.cuda(async=True), label_batch.cuda(async=True)

            data_batch, label_batch = Variable(data_batch), Variable(label_batch)
            teach_ops = [] 

            student_op = net(data_batch)
            for teach in teacher_nets:
                tops = teach(data_batch)
                tops.detach_()
                teach_ops.append(tops)
            # logging.info("Teacher ops len:{}".format(len(teach_ops)))
            loss = loss_fn_exp(student_op,teach_ops, label_batch, params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
            if i % params.save_summary_steps == 0:
                out_np = student_op.data.cpu().numpy()
                label_np = label_batch.data.cpu().numpy()
                batch_summary = {metric:metrics[metric](out_np, label_np) for metric in metrics}
                batch_summary['loss'] = loss.data[0].cpu().item()
                # print(batch_summary)
                summaries.append(batch_summary)
        loss_avg.update(loss.data[0].cpu().item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
        t.update()
    print(summaries)
    mean_metrics = {metric:np.mean([m[metric] for m in summaries]) for metric in summaries[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in mean_metrics.items())
    logging.info("Train Metrics: "+ metrics_string)
    return mean_metrics


def train_eval_experience(net, teacher_nets, train_loader, val_loader, optimizer, loss_fn_exp, metrics, params, model_dir, reporter,restore=None):
    best_val_acc = 0.0 
    if restore is not None:
        restore_file = os.path.join(args.param_path, args.resume_path + '.pth.tar')
        if not os.path.isfile(restore_file):
            logging.info("Checkpoint doesn't exist as of now")
        else:
            logging.info("Loaded checkpoints from:{}".format(restore_file))
            utils.load_checkpoint(restore_file, net, optimizer)

    # teacher_outputs = run_teacher_model(teacher_net, val_loader, params)
    decayer = newton_decayer.NewtonDecay(params.temperature[0], 1e-3) 

    for ep in range(params.num_epochs):
        logging.info("Running epoch: {}/{}".format(ep+1, params.num_epochs))
        # if ep % params.decay_epoch == 0:
        #     dtemp = decayer.decay(ep)
        #     logging.info(dtemp)
        #     params.temperature = [dtemp]
        #     logging.info("Temperature after epoch {}: {}".format(ep, dtemp))
        # train one epoch 
        train_metrics = train_experience(net, teacher_nets, train_loader, loss_fn_exp, params, metrics, optimizer)

        val_metrics = evaluate_experience(net, teacher_nets, val_loader, loss_fn_exp, params, metrics)
        reporter.report(ep, 'train_loss', train_metrics['loss'])
        reporter.report(ep, 'train_acc', train_metrics['accuracy'])
        reporter.report(ep, 'val_loss', val_metrics['loss'])
        reporter.report(ep, 'val_acc', val_metrics['accuracy'])

        val_acc = val_metrics['accuracy']
        isbest = val_acc >= best_val_acc 

        utils.save_checkpoint({"epoch":ep, "state_dict":net.state_dict(), "optimizer":optimizer.state_dict()}, 
        isBest=isbest, ckpt_dir=model_dir)
    
        if isbest:
            # if the accuracy is great  save it to best.json 
            logging.info("New best accuracy found!")
            best_val_acc = val_acc 
            best_json_path = os.path.join(model_dir, "best_model_params.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
        
        last_acc_path = os.path.join(model_dir, 'last_acc_metrics.json')
        utils.save_dict_to_json(val_metrics, last_acc_path)








if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.ParamParser(os.path.join(args.param_path, 'params.json'))
    log_path = os.path.join(args.param_path, 'log_runs')
    
    params.cuda = torch.cuda.is_available()

    utils.setLogger(os.path.join(args.param_path, "train.log"))

    logging.info("Loading the datasets")
    reporter = Reporter()
    plot_path = os.path.join(args.param_path, 'metric_plots.png')
    csv_path = os.path.join(args.param_path, "training_metrics.csv")



  
    train_loader = load_dataset('train', params)
    val_loader = load_dataset('val', params)

    logging.info("finished loading the datasets") 
    teachs = []
    
    if params.experiment_type == "experience":
        if params.student_model == "cnn":
            student_model = net.CIFARNet(params).cuda() if params.cuda else net.CIFARNet(params)
            loss_fn = net.loss_function_experience
            metrics = net.metrics
            optimizer = optim.Adam(student_model.parameters(), lr=params.learning_rate)
        if params.student_model == "cnn_2":
            student_model = net.CIFARNet2(params).cuda() if params.cuda else net.CIFARNet2(params)
            loss_fn = net.loss_function_experience
            metrics = net.metrics
            optimizer = optim.Adam(student_model.parameters(), lr=params.learning_rate)
        if params.student_model == "cnn_mnist":
            student_model = net.MNISTNet(params).cuda() if params.cuda else net.MNISTNet(params)
            loss_fn = net.loss_function_experience
            metrics = net.metrics 
            optimizer = optim.Adam(student_model.parameters(), lr=params.learning_rate)
        teachers = params.teachers 
        if "resnet18" in teachers:
            teach_resnet = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
            utils.load_checkpoint(os.path.join(os.getcwd(),'experiments/base_resnet18', 'best.pth.tar'), teach_resnet)
            teachs.append(teach_resnet)
        if "resnet18_svhn" in teachers:
            teach_resnet_svhn18 = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
            utils.load_checkpoint(os.path.join(os.getcwd(),'experiments/svhn_resnet18', 'best.pth.tar'), teach_resnet_svhn18)
            teachs.append(teach_resnet_svhn18)
        if "wresnet" in teachers:
            teach_wrn = wresnet.WideResNet(params).cuda() if params.cuda else wresnet.WideResNet(params)
            teach_wrn = nn.DataParallel(teach_wrn)
            utils.load_checkpoint(os.path.join(os.getcwd(), 'experiments/base_wresnet', 'best.pth.tar'), teach_wrn)
            teachs.append(teach_wrn)
        if "wresnet_svhn" in teachers:
            teach_wrn_svhn = wresnet.WideResNet(params).cuda() if params.cuda else wresnet.WideResNet(params)
            # teach_wrn_svhn = nn.DataParallel(teach_wrn_svhn)
            utils.load_checkpoint(os.path.join(os.getcwd(), 'experiments/wresnet_svhn', 'best.pth.tar'), teach_wrn_svhn)
            teachs.append(teach_wrn_svhn)
        if "vgg_bn" in teachers:
            vgg_bn = vggnet.vgg11_bn(params).cuda() if params.cuda else vggnet.vgg11_bn(params)
            utils.load_checkpoint(os.path.join(os.getcwd(), 'experiments/base_vggnet/best.pth.tar'),vgg_bn)
            teachs.append(vgg_bn)
        
        if "vgg" in teachers:
            vgg = vggnet.vgg11(params).cuda() if params.cuda else vggnet.vgg11_bn(params)
            #TODO: Train the vgg net without bn
        if "vgg_svhn" in teachers:
            vgg_svhn = vggnet.vgg11_bn(params).cuda() if params.cuda else vggnet.vgg11_bn(params)
            utils.load_checkpoint(os.path.join(os.getcwd(), 'experiments/vggnet_svhn/best.pth.tar'), vgg_svhn)
            teachs.append(vgg_svhn)
        

        logging.info("With alpha:{} and temperature:{}, num_epochs:{}".format(params.alpha, params.temperature, params.num_epochs))
        train_eval_experience(student_model, teachs, train_loader, val_loader, optimizer, loss_fn, metrics, params, args.param_path, reporter)
    else:
        if params.model_name == "cnn":
            model = net.CIFARNet(params).cuda() if params.cuda else net.CIFARNet(params) 
            loss_fn = net.loss_function
            metrics = net.metrics 
        elif params.model_name == "resnet18":
            model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18(params)
            loss_fn = resnet.loss_fn
            metrics = resnet.metrics
        elif params.model_name  == "wresnet":
            model = wresnet.WideResNet(params).cuda() if params.cuda else wresnet.WideResNet(params)
            model = nn.DataParallel(model) # model can't be trained on one GPU alone.
            loss_fn = wresnet.loss_fn
            metrics = wresnet.metrics
        if params.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate)
            logging.info("Selecting SGD optimizer")
        elif params.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            logging.info("Selecting Adam optimizer")
        # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate) #Change to SGD
        logging.info("Started training for {} epochs".format(params.num_epochs))
        train_and_eval(model, train_loader, val_loader, optimizer,loss_fn, metrics, params, args.param_path, reporter)
    
    # plotter = Plotter(reporter, 'epochs',['train_loss', 'val_loss', 'train_acc', ])
    entries = ['params/dataset', 'params/teachers','reporter/epoch', 'reporter/train_loss', 'reporter/train_acc', 'reporter/val_loss', 
    'reporter/val_acc', 'params/alpha', 'params/student_temperature', 'params/temperature']
    plotter = Plotter(reporter, plot_path)
    plotter.plot()
    csv_reporter = CSVReporter(reporter, params, entries, csv_path)
    csv_reporter.write_csv()




        


        
            





