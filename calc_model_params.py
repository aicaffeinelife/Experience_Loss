""" A script to count the number of trainable params in the given model """ 
import argparse 
import os 
import torch 
import utils
# from initial.distill_net import StudentNet
from models import net
from models import wide_resnet 
from models import vggnet
from models import resnet


parser = argparse.ArgumentParser()
parser.add_argument('--param_path',  help='Path to experiment folder containing the params.json')
parser.add_argument('--model_name',  help='Path to the .pth.tar of a trained model')



def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':

    args = parser.parse_args() 
    params = utils.ParamParser(os.path.join(args.param_path, 'params.json'))
    ckpt_dir = os.path.join(args.param_path, args.model_name+'.pth.tar')
    print(ckpt_dir)
    assert(os.path.isfile(ckpt_dir)), "The path:{} does not have a valid model".format(ckpt_dir)
    # assert os.path.exists(args.model_name), "The path:{} does not exist".format(args.model_name)
    # params.cuda = torch.cuda.is_available()
    params.cuda = False

    if params.experiment_type == "base":
        if params.model_name == "cnn":
            model = net.CIFARNet(params).cuda() if params.cuda else net.CIFARNet(params)
        elif params.model_name == "resnet18":
            model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
        utils.load_checkpoint(ckpt_dir, model)
        num_params = count_params(model)
        print("Experiment type:{}, Model:{}, Num Params:{}".format(params.experiment_type, params.model_name, num_params))
    elif params.experiment_type == "experience":
        if params.model_name == "cnn":
            model = net.CIFARNet(params).cuda() if params.cuda else net.CIFARNet()
        
        utils.load_checkpoint(ckpt_dir, model)
        num_params = count_params(model)
        print("Experiment type:{}, Model:{}, Num Params:{}".format(params.experiment_type, params.model_name, num_params))

    # model = net.CIFARNet(params).cuda() if params.cuda else net.CIFARNet(params)
    # utils.load_checkpoint(ckpt_dir, model)
    # num_params = count_params(model)
    # print("The model:{} has parameters:{}".format(args.model_name, num_params))
