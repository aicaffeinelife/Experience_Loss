import torch 
import os 
import torch.nn as nn 
import torch.nn.functional as F 
import argparse

from torchsummary import summary 
from models import net, wide_resnet, resnet, vggnet 
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--param_path', required=True, help='The path to param')
parser.add_argument('--model_name', required=True, help='The model name to analyze')





if __name__ == '__main__':
    args = parser.parse_args()
    param_json = os.path.join(args.param_path, 'params.json')
    params = utils.ParamParser(param_json)
    params.cuda = torch.cuda.is_available()
    model = None

    if args.model_name == "net":
        model = net.CIFARNet(params).cuda() if params.cuda else net.CIFARNet(params)     
    elif args.model_name == "resnet18":
        model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
    elif args.model_name == "wrn":
        model = wide_resnet.WideResNet(params).cuda() if params.cuda else wide_resnet.WideResNet(params)
    elif args.model_name == "vgg":
        model = vggnet.vgg11_bn(params).cuda() if params.cuda else vggnet.vgg11_bn(params)
    # elif args.model_name == "net2":
    #     model = net.CIFARNet2(params).cuda() if params.cuda else net.CIFARNet2(params)
    
    summary(model, (3,32,32))