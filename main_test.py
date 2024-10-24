import argparse
import copy
import json
import os
import sys

import torch

import attack
import data_loaders
from functions import *
from models import *
from utils import val, val_success_rate,dif_visul_val
from functools import partial
import seaborn as sns

parser = argparse.ArgumentParser()
# just use default setting
parser.add_argument('-j','--workers',default=0, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=128, type=int,metavar='N',help='mini-batch size')
parser.add_argument('-sd', '--seed',default=42,type=int,help='seed for initializing training.')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset', default='cifar10',type=str,help='dataset')
parser.add_argument('-arch','--model', default='vgg11', type=str,help='model')
parser.add_argument('-T','--time', default=8, type=int, metavar='N',help='snn simulation time, set 0 as ANN')
parser.add_argument('-tau','--tau',default=1., type=float,metavar='N',help='leaky constant')
parser.add_argument('-id', '--identifier', type=str, help='model statedict identifier')
parser.add_argument('-config', '--config', default='', type=str,help='test configuration file')
parser.add_argument('-en', '--encode', default='constant', type=str, help='poisson constant')

##ft


# training configuration
parser.add_argument('-dev','--device',default='0',type=str,help='device')

# adv atk configuration
parser.add_argument('-atk','--attack',default='', type=str, help='attack')
parser.add_argument('-eps','--eps',default=8, type=float, metavar='N', help='attack eps')
parser.add_argument('-atk_m','--attack_mode', default='', type=str, help='attack mode')

# only pgd
parser.add_argument('-alpha', '--alpha',default=2.55/1,type=float,metavar='N',help='pgd attack alpha')
parser.add_argument('-steps', '--steps',default=7,type=int,metavar='N',help='pgd attack steps')
parser.add_argument('-bb', '--bbmodel',default='',type=str,help='black box model') #
parser.add_argument('-stdout', '--stdout',default='',type=str,help='log file')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    dvs = False
    if args.dataset.lower() == 'cifar10':
        use_cifar10 = True
        num_labels = 10
        H,W =32,32
    elif args.dataset.lower() == 'cifar100':
        use_cifar10 = False
        num_labels = 100
        H,W =32,32
    elif args.dataset.lower() == 'svhn':
        num_labels = 10
    elif args.dataset.lower() == 'dvscifar':
        num_labels = 10
        assert args.time == 10
        dvs = True
    elif args.dataset.lower() == 'dvsgesture':
        num_labels = 11
        assert args.time == 10
        dvs = True
        init_s = 48
    elif args.dataset.lower() == 'nmnist':
        num_labels = 10
        assert args.time == 10
        dvs = True
        init_s = 34
    elif args.dataset.lower() == 'tinyimagenet':
        num_labels = 200
        H,W =64,64

    if args.time != 0:
        freq_filter = make_filter_0(H, W, filter_windows=[16,14,12,10,8,6,4,2]) 
    else:
        freq_filter = 1

    log_dir = 'logs/%s-results-epoch300'% (args.dataset)

    model_dir = 'logs/%s-checkpoints-epoch300'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = get_logger(os.path.join(log_dir, '%s.log'%(args.identifier+args.suffix)))
    logger.info('start testing!')

    seed_all(args.seed)
    if 'dvsgesture' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvsgesture(root='/home/datasets/DVSGesture/')
    elif 'dvscifar' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvscifar(root='/home/datasets/CIFAR10DVS/')
    elif 'nmnist' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_nmnist(root='/home/datasets/NMNIST/')
    elif 'cifar' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_cifar(use_cifar10=use_cifar10)
    elif 'tinyimagenet' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_tinyimagenet(root='/home/dataset/tiny-imagenet-200')
    elif args.dataset.lower() == 'svhn':
        train_dataset, val_dataset, znorm = data_loaders.build_svhn()
    else:
        raise AssertionError("data not supported")

    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    if 'cnndvs' in args.model.lower():
        model = CNNDVS(args.time, num_labels, args.tau, 2, init_s)
    elif 'vggdvs' in args.model.lower():
        model = VGGDVS(args.model.lower(), args.time, num_labels, znorm, args.tau)
    elif 'vgg' in args.model.lower():
        model = VGG_TAU(args.model.lower(), args.time, num_labels, znorm, args.tau, args=args)
    elif 'resnet19' in args.model.lower():
        model = ResNet19Tau(args.time, args.tau, num_labels, znorm,args=args)
    elif 'wideresnet' in args.model.lower():
        model = WideResNetTau(args.model.lower(), args.time, num_labels, znorm,args=args)
    else:
        raise AssertionError("model not supported")

    model = nn.DataParallel(model) 
    model.module.set_simulation_time(args.time) 

    state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    if len(args.bbmodel) > 0:
        bbmodel = copy.deepcopy(model)
        bbstate_dict = torch.load(os.path.join(model_dir, args.bbmodel+'.pth'), map_location=torch.device('cpu'))
        bbmodel.load_state_dict(bbstate_dict, strict=False)
        bbmodel.poisson = (args.encode.lower() == 'poisson')
    else:
        bbmodel = None

    if len(args.config) > 0:
        with open(args.config+'.json', 'r') as f:
            config = json.load(f)
    else:
        config = [{}]
    
    for atk_config in config:
        for arg in atk_config.keys():
            setattr(args, arg, atk_config[arg])
        if 'bb' in atk_config.keys() and atk_config['bb']:
            atkmodel = bbmodel
        else:
            atkmodel = model

        if args.attack_mode == 'bptt':
            ff = partial(BPTT_attack, freq_filter)
        elif args.attack_mode == 'bptr':
            ff = BPTR_attack
        else:
            ff = Act_attack

        if args.attack.lower() == 'fgsm':
            atk = attack.FGSM(atkmodel, forward_function=ff, eps=args.eps / 255, T=args.time)
        elif args.attack.lower() == 'pgd':
            atk = attack.PGD(atkmodel, forward_function=ff, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps, T=args.time)
        elif args.attack.lower() == 'bim':
            atk = attack.BIM(atkmodel, forward_function=ff, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps, T=args.time)
        elif args.attack.lower() == 'cw':
            atk = attack.CW(atkmodel, forward_function=ff, T=args.time)
        elif args.attack.lower() == 'gn':
            atk = attack.GN(atkmodel, forward_function=ff, eps=args.eps / 255, T=args.time)
        else:
            atk = None
        
        
        if atk is not None:
            acc = val(model, test_loader, device, args.time, dvs, atk,freq_filter)
            logger.info('%s'%(args.attack)+'-acc:{:.3f}'.format(acc))
        else:
            acc = val(model, test_loader, device, args.time, dvs, atk,freq_filter)
            logger.info('clean-acc:{:.3f}/'.format(acc))



if __name__ == "__main__":
    main()