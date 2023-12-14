################IJCNN用にResNet-50とWide　ResNetを学習できるように変更     2023/01/11   #####################
#####実装できたらもとのパイソンファイルに上書きする？#############

####main_resnet_new.pyの学習が終わったら、名前を逆にする。こっちが元のやつなので

#Essential Imports
import os
import sys
import argparse
import datetime
import time
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import random
# import torchsummary
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd.function import Function

###umap###
import umap
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sklearn.datasets
##########

import matplotlib.pyplot as plt
from torch.utils.data.dataset import Subset
from scipy.stats import entropy
from torch.optim import lr_scheduler

###.pyからのインポート###
#from utils import AverageMeter
from proximity import Proximity
from contrastive_proximity import Con_Proximity
from resnet_model_IJCNN import *  # Imports the ResNet Model

# #####deep metric laerning#####
# #pip install pytorch-metric-learning
# from pytorch_metric_learning.miners import TripletMarginMiner
# from pytorch_metric_learning.distances import CosineSimilarity
# from pytorch_metric_learning.losses import TripletMarginLoss
# from pytorch_metric_learning.reducers import ThresholdReducer






parser = argparse.ArgumentParser("CNN")
parser.add_argument('--n_layers', type=int, default=20, required=True, help="resnet layer num : 20(basic) or 56(bottleneck) or 28(wide resnet)") #20(basic) or 56(bottleneck) or 28(wide resnet)
parser.add_argument('--use_cot', action='store_true')
parser.add_argument('--use_mixup', action='store_true')
parser.add_argument('--dataset', type=str, required=True) #CIFAR-10, CIFAR-100, Tiny-ImageNet, ImageNet-1k, SVHN, food-101
parser.add_argument('--loss_type', type=str, required=True) #ce center pc
parser.add_argument('--use_pretrain', action='store_true', help='False:pretraining, True:training with pretrain model')
parser.add_argument('--batch_size', type=int, default=128, required=True, help="mini batch size (32, 64, 128, 256, 512)")
parser.add_argument('--seed', type=int, default=42, required=True, help="seed value")
parser.add_argument('--option', type=str, default='')
parser.add_argument('--CUDA_num', type=int, default=0, required=True, help="cuda number (0,1,2,3,4,5,6,7)") 
parser.add_argument('--CUDA_num2', type=int, default=0, help="cuda number (0,1,2,3,4,5,6,7)") 
parser.add_argument('--CUDA_num3', type=int, default=0, help="cuda number (0,1,2,3,4,5,6,7)") 
parser.add_argument('--lr_model', type=float, default=0.1, help="learning rate for CE Loss")
parser.add_argument('--lr_prox', type=float, default=0.5, help="learning rate for Proximity Loss") # as per paper
parser.add_argument('--lr_conprox', type=float, default=0.0001, help="learning rate for Con-Proximity Loss") # as per paper
args = parser.parse_args()


if args.use_pretrain == False:
    # assert not(args.loss_type == 'center') and not(args.loss_type == 'pc'), 'Center or PC must be use_pretrain True'
    assert args.loss_type == 'ce', 'Loss function for feature space must be use_pretrain True'
    assert args.use_cot == False, 'COT must be use_pretrain True'
    


print("#######################################################################################################################")
print(f"cot : {args.use_cot}\n mixup : {args.use_mixup}\n dataset : {args.dataset}\n loss : {args.loss_type}\n  batch_size : {args.batch_size}\n lr_model : {args.lr_model}\n lr_prox : {args.lr_prox}\n lr_conprox : {args.lr_conprox}\n")
#--use_cot --use_mixup --dataset --loss_type --use_pretrain --batch_size --seed --option --CUDA_num --lr_model --lr_prox --lr_conprox
#--dataset --loss_type --use_pretrain --batch_size --seed --option --CUDA_num
##/ya##print("#######################################################################################################################")
##/ya##print("--lr_model", args.lr_model)
#####################################各種設定###########################################################################

cot_bool = 'COT-' if args.use_cot else ''
mixup_bool = 'M' if args.use_mixup else ''
svhn_bool = '[SVHN]' if args.dataset == 'SVHN' else ''  #CIFAR10とSVHNのクラス数が被ってしまったため，このようにして区別（応急処置）

#network architecture (''=Basic)
if args.n_layers==20:
    architecture = ''
elif args.n_layers==56:
    architecture = 'bottleneck'
elif args.n_layers==28:
    architecture = 'wideResNet'
    

    
if args.dataset == 'CIFAR-10' or args.dataset == 'SVHN':
    dataset_name = '10'
elif args.dataset == 'CIFAR-100':
    dataset_name = '100'
elif args.dataset == 'Tiny-ImageNet':
    dataset_name = '200'
elif args.dataset == 'ImageNet-1k':
    dataset_name = '1000'
elif args.dataset == 'ImageNet-21k':
    dataset_name = '21000'
elif args.dataset == 'food-101':
    dataset_name = '101'


#####保存先モデル#####
if args.use_pretrain:
    #trained_model = './trained_models2/' + mixup_bool + dataset_name + 'pre_' + str(args.batch_size) + 'batch(' + str(args.seed) + ')' + args.option + '.pth'
    #同じ事前学習を使って学習するようのやつ（終わったら消す？）
    trained_model = './trained_models[ResNet]/' + mixup_bool + svhn_bool + dataset_name + 'pre_128batch(' + str(args.seed) + ')' + architecture + args.option + '.pth'
    
    model_name = cot_bool + mixup_bool + svhn_bool + dataset_name + args.loss_type + '_' + str(args.batch_size) + 'batch(' + str(args.seed) + ')' + architecture + args.option
else:
    model_name = mixup_bool + svhn_bool + dataset_name + 'pre_' + str(args.batch_size) + 'batch(' + str(args.seed) + ')' + architecture + args.option


keep_model = './trained_models[ResNet]/' + model_name + '.pth'
best_model = './trained_models[ResNet]/BestAcc_' + model_name + '.pth'

best_model_name = 'BestAcc_' + model_name

#バッチサイズごとにエポック数を設定することでイテレーションを合わせる
max_num=200

if args.use_pretrain:
    if args.batch_size == 32:
        epoch_num = 50
    elif args.batch_size == 64:
        epoch_num = 100
    elif args.batch_size == 128: #base
        epoch_num = 200
    elif args.batch_size == 256:
        epoch_num = 400
    elif args.batch_size == 512:
        epoch_num = 800
else:
    if args.batch_size == 32:
        epoch_num = 25
    elif args.batch_size == 64:
        epoch_num = 50
    elif args.batch_size == 128: #base
        epoch_num = 100
    elif args.batch_size == 256:
        epoch_num = 200
    elif args.batch_size == 512:
        epoch_num = 400
 
        
if args.use_pretrain:
    print("trained_model", trained_model)
print("save model", keep_model)

print(f'batch size = {args.batch_size}, epoch = {epoch_num}')
#######################################################################################################################



# GPUの確認
use_gpu = torch.cuda.is_available()
print('Use CUDA:', use_gpu, args.CUDA_num)


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    print("seed値は{}".format(seed))

torch_fix_seed(args.seed)

print(torchvision.datasets)


#####################################データセットの設定####################################################################
if args.dataset == 'SVHN': #左右反転を使わないため区別
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=1),
                                          transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    
elif args.dataset == 'food-101':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) # 正規化定数
    
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=1),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize
                                      ])
    transform_test = transforms.Compose([transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize
                                      ])

else:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=1),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

if args.dataset == 'CIFAR-10':
    train_data = torchvision.datasets.CIFAR10(root="./", train=True, transform=transform_train, download=True)
    test_data = torchvision.datasets.CIFAR10(root="./", train=False, transform=transform_test, download=True)
elif args.dataset == 'CIFAR-100':
    train_data = torchvision.datasets.CIFAR100(root="./", train=True, transform=transform_train, download=True)
    test_data = torchvision.datasets.CIFAR100(root="./", train=False, transform=transform_test, download=True)
elif args.dataset == 'Tiny-ImageNet':
    dataset_dir = os.path.join(os.getcwd(), 'tiny-imagenet-200')
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])  #正規化定数
    train_data = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomHorizontalFlip(),      # ランダムに画像をフリップ（水増し）
                transforms.ToTensor(),
                normalize
            ]))
    
    test_data = datasets.ImageFolder(
            val_dir,
            transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]))
elif args.dataset == 'ImageNet-1k':
    dataset_dir = os.path.join(os.getcwd(), 'ImageNet/ILSVRC2012/')
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) # 正規化定数

    train_data = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),      # ランダムに画像をフリップ（水増し）
                transforms.ToTensor(),
                normalize
            ]))
    
    test_data = datasets.ImageFolder(
            val_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]))#elif dataset == 'ImageNet-21k':
elif args.dataset == 'SVHN':
    train_data = torchvision.datasets.SVHN(root="./", split="train", transform=transform_train, download=True)
    test_data = torchvision.datasets.SVHN(root="./", split="test", transform=transform_test, download=True)
    
elif args.dataset == 'food-101':
    train_data = torchvision.datasets.Food101(root="./", split="train", transform=transform_train, download=True)
    test_data = torchvision.datasets.Food101(root="./", split="test", transform=transform_test, download=True)
    

print(train_data)
print(test_data)


#####Tiny Imagenetのクラス名を取得(get_class_name)#####
if args.dataset == 'Tiny-ImageNet':
    class_to_name = dict()
    fp = open(os.path.join(os.getcwd(), 'tiny-imagenet-200', 'words.txt'), 'r')
    data = fp.readlines()
    for line in data:
        words = line.strip('\n').split('\t')
        class_to_name[words[0]] = words[1].split(',')[0]
    fp.close()

    #print(class_to_name)
    
    
    
if args.dataset == 'Tiny-ImageNet':
    idx_to_class = {i: c for c, i in test_data.class_to_idx.items()}
    #for i in range(200):
        #print(i, class_to_name[idx_to_class[i]])

#######################################################################################################################








#######################################ネットワークの構築###################################################################
# ResNetの層数を指定 (e.g. 20, 32, 44, 47, 56, 110, 1199)
n_layers = args.n_layers
device = torch.device("cuda:{}".format(args.CUDA_num) if use_gpu else "cpu")        
num_classes = int(dataset_name)


if args.n_layers==20:
    if args.dataset == 'CIFAR-10' or args.dataset == 'SVHN':
        feat_dim = 256
    elif args.dataset == 'CIFAR-100':
        feat_dim = 256
    elif args.dataset == 'Tiny-ImageNet':
        feat_dim = 512
    elif args.dataset == 'ImageNet-1k' or args.dataset == 'food-101':
        feat_dim = 512
        
elif args.n_layers==56: #bottleneck
    if args.dataset == 'CIFAR-10' or args.dataset == 'SVHN':
        feat_dim = 1024 #256*4
    elif args.dataset == 'CIFAR-100':
        feat_dim = 1024 #256*4
    elif args.dataset == 'Tiny-ImageNet':
        feat_dim = 2048 #512*4
    elif args.dataset == 'ImageNet-1k' or args.dataset == 'food-101':
        feat_dim = 2048 #512*4
        
elif args.n_layers==28: #wide resnet
    if args.dataset == 'CIFAR-10' or args.dataset == 'SVHN':
        feat_dim = 640
    elif args.dataset == 'CIFAR-100':
        feat_dim = 640
    elif args.dataset == 'Tiny-ImageNet':
        feat_dim = 640
    elif args.dataset == 'ImageNet-1k' or args.dataset == 'food-101':
        feat_dim = 640
        
# ResNetを構築
if args.n_layers==20:
    model = ResNetBasicBlock(depth=n_layers, n_class=int(dataset_name))    # BasicBlock構造を用いる場合
elif args.n_layers==56:
    model = ResNetBottleneck(depth=n_layers, n_class=int(dataset_name)) 
elif args.n_layers==28:
    model = Wide_ResNet(28, 10, 0.3, int(dataset_name))
        
if use_gpu:
    if args.dataset == 'ImageNet-1k':# or args.dataset == 'food-101':
        print("#####Use DataParallel#####")
        model = torch.nn.DataParallel(model, device_ids=[args.CUDA_num, args.CUDA_num2, args.CUDA_num3]).cuda(args.CUDA_num)
    else:
        model.cuda(args.CUDA_num)
    

if args.use_pretrain:
    #モデルの読み込み
    checkpoint = torch.load(trained_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("このモデルのseed値は{}です".format(checkpoint['seed']))

    #for logs in chekpoint['log']:
     #   print(logs)

    #model.state_dict()

# モデルの情報を表示
#torchsummary.summary(model, (3, 32, 32))
#######################################################################################################################




# データローダーの準備
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)
# ネットワークを評価モードへ変更
model.eval()

# 評価の実行
count = 0
with torch.no_grad():
    for image, label in test_loader:

        if use_gpu:
            image = image.cuda(args.CUDA_num)
            label = label.cuda(args.CUDA_num)

        pp, y = model(image)

        pred = torch.argmax(y, dim=1)
        count += torch.sum(pred == label)

print("test accuracy: {}".format(count.item() / len(test_data)))





###################optimizer等の設定###################################################################################
##########COT適用################
# if args.use_cot:
#     criterion_complement = ComplementEntropy(num_classes=num_classes, CUDA_num=args.CUDA_num).to(device)
#     optimizer_complement = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-04, momentum=0.9)
    
    

#########Center Loss, PC Lossを1箇所に導入##############
if args.loss_type == 'ce':
    criterion_nllloss = nn.NLLLoss().to(device) #CrossEntropyLoss = log_softmax + NLLLoss

elif args.loss_type == 'center':
    criterion_nllloss = nn.NLLLoss().to(device) #CrossEntropyLoss = log_softmax + NLLLoss
    criterion_prox_64 = Proximity(num_classes=num_classes, feat_dim=feat_dim, CUDA_num=args.CUDA_num, use_gpu=use_gpu)

elif args.loss_type == 'pc':
    criterion_nllloss = nn.NLLLoss().to(device) #CrossEntropyLoss = log_softmax + NLLLoss
    criterion_prox_64 = Proximity(num_classes=num_classes, feat_dim=feat_dim, CUDA_num=args.CUDA_num, use_gpu=use_gpu)
    criterion_conprox_64 = Con_Proximity(num_classes=num_classes, feat_dim=feat_dim, CUDA_num=args.CUDA_num, use_gpu=use_gpu)
    



    
if args.loss_type == 'ce':
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=1e-04, momentum=0.9)

elif args.loss_type == 'center':
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=1e-04, momentum=0.9)
    optimizer_prox_64 = torch.optim.SGD(criterion_prox_64.parameters(), lr=args.lr_prox)

elif args.loss_type == 'pc':
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=1e-04, momentum=0.9)
    optimizer_prox_64 = torch.optim.SGD(model.parameters(), lr=args.lr_prox)
    optimizer_conprox_64 = torch.optim.SGD(criterion_conprox_64.parameters(), lr=args.lr_conprox)
    
    
    
if args.dataset == 'CIFAR-10' or args.dataset == 'SVHN':
    weight_prox = 1
    weight_conprox = 0.0001
elif args.dataset == 'CIFAR-100' or args.dataset == 'food-101':
    weight_prox = 0.1
    weight_conprox = 0.00001
elif args.dataset == 'Tiny-ImageNet':
    weight_prox = 0.05
    weight_conprox = 0.000005
elif args.dataset == 'ImageNet-1k':
    weight_prox = 0.01
    weight_conprox = 0.000001
    
print_freq = 1000000000
#######################################################################################################################





###################################追加学習時のみスケジューラを導入（事前学習時はスケジューラなし）###################################
if args.use_pretrain:
    #エポック数の50％と75％で減衰
    sche1 = int(epoch_num*0.5)
    sche2 = int(epoch_num*0.75)
    scheduler = [sche1, sche2]

    #########Center Loss, PC Lossを1箇所に導入##############
    adjust_learning_rate = lr_scheduler.MultiStepLR(optimizer_model, scheduler, gamma=0.1)
    if args.loss_type == 'center':
        adjust_learning_rate_prox = lr_scheduler.MultiStepLR(optimizer_prox_64, scheduler, gamma=0.1)
    elif args.loss_type == 'pc':
        adjust_learning_rate_prox = lr_scheduler.MultiStepLR(optimizer_prox_64, scheduler, gamma=0.1)
        adjust_learning_rate_conprox = lr_scheduler.MultiStepLR(optimizer_conprox_64, scheduler, gamma=0.1)
        
    # if args.use_cot:
    #     adjust_learning_rate_complement = lr_scheduler.MultiStepLR(optimizer_complement, scheduler, gamma=0.1)

#######################################################################################################################





############################学習#######################################################################################
n_iter = len(train_data) / args.batch_size
#データローダーの設定
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)#2 #pin_memory：メモリ領域がページングされず高速化
print("CPU num", os.cpu_count())

# xent_losses = AverageMeter() #Computes and stores the average and current value 計算するための処理
# prox_losses_64 = AverageMeter()
# conprox_losses_64 = AverageMeter()
# losses = AverageMeter()

epoch_list = []
mean_loss_list = []
mean_accuracy_list = []
val_accuracy_list = []
val_loss_list = []
log = []
val_all_acc = []
best_acc = 0.0

torch_fix_seed(args.seed)
start = time.time()
for epoch in range(1, epoch_num+1):
    # ネットワークを学習モードへ変更
    model.train()
    sum_loss = 0.0
    sum_loss_val = 0.0
    count = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):     
        if use_gpu: 
            data, labels = data.cuda(args.CUDA_num), labels.cuda(args.CUDA_num)
        batch = data.size(0)
        
        #print("###############################################################################")
        #for i in range(len(labels)):
            #print(class_to_name[idx_to_class[int(labels[i])]])
            
        #########Center Loss, PC Lossを1箇所に導入##############
        if args.use_mixup:
            lam = np.random.beta(0.3, 0.3)
            rand_batch = torch.randperm(batch)
            data = lam * data + (1 - lam) * data[rand_batch]
            feats, outputs = model(data)    
            loss_xent = lam * criterion_nllloss(outputs, labels) + (1 - lam) * criterion_nllloss(outputs, labels[rand_batch])
        else:
            feats, outputs = model(data)
            loss_xent = criterion_nllloss(outputs, labels)
 
                

        if args.loss_type == 'center':
            if args.use_mixup:
                loss_prox_64 = lam * criterion_prox_64(feats, labels) + (1 - lam) * criterion_prox_64(feats, labels[rand_batch])
            else:
                loss_prox_64 = criterion_prox_64(feats, labels)                    
            loss_prox_64 *= weight_prox
            loss = loss_xent + loss_prox_64

        elif args.loss_type == 'pc':
            if args.use_mixup:
                loss_prox_64 = lam * criterion_prox_64(feats, labels) + (1 - lam) * criterion_prox_64(feats, labels[rand_batch])
                loss_conprox_64 = lam * criterion_conprox_64(feats, labels) + (1 - lam) * criterion_conprox_64(feats, labels[rand_batch])
            else:
                loss_prox_64 = criterion_prox_64(feats, labels) 
                loss_conprox_64 = criterion_conprox_64(feats, labels)                
            loss_prox_64 *= weight_prox  
            loss_conprox_64 *= weight_conprox
            loss = loss_xent + loss_prox_64 - loss_conprox_64
                    

        else:
            loss = loss_xent
 



        optimizer_model.zero_grad()
        if args.loss_type == 'center':
            optimizer_prox_64.zero_grad()
        elif args.loss_type == 'pc':
            optimizer_prox_64.zero_grad()
            optimizer_conprox_64.zero_grad()

        loss.backward()

        optimizer_model.step() 
        if args.loss_type == 'center':
            for param in criterion_prox_64.parameters():
                param.grad.data *= (1. / weight_prox)
            optimizer_prox_64.step() 
        elif args.loss_type == 'pc':
            for param in criterion_prox_64.parameters():
                param.grad.data *= (1. / weight_prox)
            optimizer_prox_64.step() 

            for param in criterion_conprox_64.parameters():
                param.grad.data *= (1. / weight_conprox)
            optimizer_conprox_64.step() 


        # losses.update(loss.item(), labels.size(0)) 
        # xent_losses.update(loss_xent.item(), labels.size(0))        
        # if args.loss_type == 'center':
        #     prox_losses_64.update(loss_prox_64.item(), labels.size(0))
        # elif args.loss_type == 'pc':
        #     prox_losses_64.update(loss_prox_64.item(), labels.size(0))
        #     conprox_losses_64.update(loss_conprox_64.item(), labels.size(0))
                
        sum_loss += loss.item()

        pred = torch.argmax(outputs, dim=1)
        if args.use_mixup:
            count += (lam * torch.sum(pred == labels)) + ((1 - lam) * torch.sum(pred == labels[rand_batch]))
        else:
            count += torch.sum(pred == labels)
        
        ##################COT###################
        if args.use_cot:
            feats, outputs = model(data)
            loss = criterion_complement(outputs, labels)
            cotloss = loss
            optimizer_complement.zero_grad()
            loss.backward()
            optimizer_complement.step()
            
    #####検証########################################################################
    # ネットワークを評価モードへ変更
    model.eval()
    
    # 評価の実行
    count_val = 0
    with torch.no_grad():
        for image, label in test_loader:
            if use_gpu:
                image = image.cuda(args.CUDA_num)
                label = label.cuda(args.CUDA_num)
            xx, yy = model(image)
            
            #######Loss culculate######
            loss_xent_val = criterion_nllloss(yy, label)
            if args.loss_type == 'center':
                loss_prox_64_val = criterion_prox_64(xx, label)                    
                loss_prox_64_val *= weight_prox
                
                loss_val = loss_xent_val + loss_prox_64_val
                
            elif args.loss_type == 'pc':
                loss_prox_64_val = criterion_prox_64(xx, label) 
                loss_conprox_64_val = criterion_conprox_64(xx, label)                
                loss_prox_64_val *= weight_prox  
                loss_conprox_64_val *= weight_conprox
                
                loss_val = loss_xent_val + loss_prox_64_val - loss_conprox_64_val

            else:
                loss_val = loss_xent_val
            ###########################
            
            pred = torch.argmax(yy, dim=1)
            count_val += torch.sum(pred == label)
            
            sum_loss_val += loss_val.item()
    ###############################################################################

                
             
    #スケジューラ
    if args.use_pretrain:
        #########Center Loss, PC Lossを1箇所に導入##############
        adjust_learning_rate.step()
        if args.loss_type == 'center':
            adjust_learning_rate_prox.step()
        elif args.loss_type == 'pc':
            adjust_learning_rate_prox.step()
            adjust_learning_rate_conprox.step()

        # if args.use_cot:
        #     adjust_learning_rate_complement.step()
            

 #学習曲線プログラム
    epoch_list.append(epoch)
    mean_loss_list.append(sum_loss / n_iter)
    val_loss_list.append(sum_loss_val / n_iter)
    mean_accuracy_list.append(count.item() / len(train_data))
    val_accuracy_list.append(count_val.item() / len(test_data))
    
    
    #######保存用（エポック・認識精度）#######
    val_all_acc.append([epoch, count_val.item() / len(test_data)]) #２次元配列
    

    log.append("epoch: {}\nmean loss: {}\nmean  acc: {}\nval  loss: {}\nval   acc: {}\nelap_time: {}\n".format(epoch,
                                                                                 sum_loss / n_iter,
                                                                                 count.item() / len(train_data),
                                                                                 sum_loss_val / n_iter,
                                                                                 count_val.item() / len(test_data),
                                                                                 time.time() - start))

    print("epoch: {}\nmean loss: {}\nmean  acc: {}\nval  loss: {}\nval   acc: {}\nelap_time: {}\n".format(epoch,
                                                                                 sum_loss / n_iter,
                                                                                 count.item() / len(train_data),
                                                                                 sum_loss_val / n_iter,
                                                                                 count_val.item() / len(test_data),
                                                                                 time.time() - start))

    #####save best acc model####
    if best_acc <= (count_val.item() / len(test_data)):
        best_acc = (count_val.item() / len(test_data))
        torch.save({
                'model_state_dict' : model.state_dict(),
                'seed' : args.seed,
                'log' : log,
                'args' : args,
                'epoch_list' : epoch_list,
                'mean_loss_list' : mean_loss_list,
                'val_loss_list' : val_loss_list,
                'mean_accuracy_list' : mean_accuracy_list,
                'val_accuracy_list' : val_accuracy_list,
                }, best_model)
        
    ####save model every 10epochs for visualizing UMAP#####
    # if epoch % 10 == 0 or epoch == 1:
    #     every_model = './trained_models[ResNet]/EPOCH-' + str(epoch) + '_' + model_name + '.pth'
    #     torch.save({
    #             'model_state_dict' : model.state_dict(),
    #             'seed' : args.seed,
    #             'resnet_layer' : args.n_layers,
    #             'log' : log,
    #             'args' : args,
    #             }, every_model)
#######################################################################################################################





# データローダーの準備
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)
# ネットワークを評価モードへ変更
model.eval()

# 評価の実行
count = 0
with torch.no_grad():
    for image, label in test_loader:

        if use_gpu:
            image = image.cuda(args.CUDA_num)
            label = label.cuda(args.CUDA_num)
            
        pp, y = model(image)

        pred = torch.argmax(y, dim=1)
        count += torch.sum(pred == label)

print("test accuracy: {}".format(count.item() / len(test_data)))



plt_path = './learning_curve[ResNet]/' + model_name + '_loss.svg'
plt.figure()
plt.plot(epoch_list, mean_loss_list, label='loss (train)')
plt.plot(epoch_list, val_loss_list, label='loss (valid)')
plt.xlabel("epoch")     # x軸ラベル
plt.ylabel("loss")      # y軸ラベル
plt.legend()            # 凡例
plt.savefig(plt_path)
plt.show()

plt_path = './learning_curve[ResNet]/' + model_name + '_acc.svg'
plt.figure()
plt.plot(epoch_list, mean_accuracy_list, label='accuracy (train)')
plt.plot(epoch_list, val_accuracy_list, label='accuracy (valid)')
plt.xlabel("epoch")     # x軸ラベル
plt.ylabel("accuracy")      # y軸ラベル
plt.legend()            # 凡例
#plt.ylim(0.0, 1.0)
plt.savefig(plt_path)
plt.show()





#モデルの保存
torch.save({
            'model_state_dict' : model.state_dict(),
            'seed' : args.seed,
            'resnet_layer' : args.n_layers,
            'log' : log,
            'args' : args,
            }, keep_model)
#model.state_dict()

###################################特徴空間スコア##########################################################################
# データセットごとの画素数
if args.dataset == 'CIFAR-10' or args.dataset == 'CIFAR-100' or args.dataset == 'SVHN':
    num_pix = 1024
    row_col_pix = 32
    #画像数
    total_img = 10000
elif args.dataset == 'Tiny-ImageNet':
    num_pix = 4096
    row_col_pix = 64
    #画像数
    total_img = 10000
elif args.dataset == 'ImageNet-1k' or args.dataset == 'food-101':
    num_pix = 50176
    row_col_pix = 224
    #画像数
    total_img = 50000

    
###このモデル用のディレクトリを作成###
#mkdir = "./UMAP/3d_"+model_name
#os.makedirs(mkdir, exist_ok=True)




#syuusei sita
# 評価のために再度ResNetを構築
if args.n_layers==20:
    model = ResNetBasicBlock_oneoutput(depth=args.n_layers, n_class=int(dataset_name)) # BasicBlock構造を用いる場合
elif args.n_layers==56:
    model = ResNetBottleneck_oneoutput(depth=args.n_layers, n_class=int(dataset_name)) 
elif args.n_layers==28:
    model = Wide_ResNet_oneoutput(28, 10, 0.3, int(dataset_name))

if use_gpu:
    model.cuda(args.CUDA_num)

#モデルの読み込み
checkpoint = torch.load(keep_model)
model.load_state_dict(checkpoint['model_state_dict'])
print("このモデルのseed値は{}です".format(checkpoint['seed']))

#for logs in checkpoint['log']:
    #print(logs)
        
#model.state_dict()

# モデルの情報を表示
#torchsummary.summary(model, (3, 32, 32))





# データローダーの準備
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

# ネットワークを評価モードへ変更
model.eval()

# 評価の実行
count = 0
with torch.no_grad():
    for image, label in test_loader:

        if use_gpu:
            image = image.cuda(args.CUDA_num)
            label = label.cuda(args.CUDA_num)
            
        y = model(image)
        pred = torch.argmax(y, dim=1)
        count += torch.sum(pred == label)

#test_acc = count.item() / len(test_data)
print("test accuracy: {}".format(count.item() / len(test_data)))

############################best model eval#################################################
# 評価のために再度ResNetを構築
if args.n_layers==20:
    model = ResNetBasicBlock_oneoutput(depth=args.n_layers, n_class=int(dataset_name)) # BasicBlock構造を用いる場合
elif args.n_layers==56:
    model = ResNetBottleneck_oneoutput(depth=args.n_layers, n_class=int(dataset_name)) 
elif args.n_layers==28:
    model = Wide_ResNet_oneoutput(28, 10, 0.3, int(dataset_name))
        
if use_gpu:
    model.cuda(args.CUDA_num)

#モデルの読み込み
checkpoint = torch.load(best_model)
model.load_state_dict(checkpoint['model_state_dict'])
print("このモデルのseed値は{}です".format(checkpoint['seed']))
# for logs in checkpoint['log']:
#     print(logs)
        


# データローダーの準備
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

# ネットワークを評価モードへ変更
model.eval()

# 評価の実行
count = 0
with torch.no_grad():
    for image, label in test_loader:

        if use_gpu:
            image = image.cuda(args.CUDA_num)
            label = label.cuda(args.CUDA_num)
            
        y = model(image)
        pred = torch.argmax(y, dim=1)
        count += torch.sum(pred == label)

#test_acc = count.item() / len(test_data)
print("test accuracy: {}".format(count.item() / len(test_data)))

