################IJCNN用にResNet-50とWide　ResNetを学習できるように変更     2023/01/11   #####################
#####実装できたらもとのパイソンファイルに上書きする？#############

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np


from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

##############################Wide ResNet################################################################################################

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        
        if num_classes == 200:
            self.avgpool = nn.AvgPool2d(16)  #o応急処置　　TIny iMageNetの学習が終わったら，修正する（8でも16でもいけるように） 2023/01/13
        else:
            self.avgpool = nn.AvgPool2d(8)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
                #print("out shape",out.shape) #torch.Size([100(batchsize), 640(conv channel), 8, 8])
        out = F.relu(self.bn1(out))
        #out = F.avg_pool2d(out, 8)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        y = self.linear(out)

        return out, F.log_softmax(y, dim=1)

    
    
class Wide_ResNet_oneoutput(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_oneoutput, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        
        if num_classes == 200:
            self.avgpool = nn.AvgPool2d(16)  #o応急処置　　TIny iMageNetの学習が終わったら，修正する（8でも16でもいけるように） 2023/01/13
        else:
            self.avgpool = nn.AvgPool2d(8)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        #out = F.avg_pool2d(out, 8)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
############################################################################################################################################
















############################Bottleneck ResNet##################################################################################################

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),   #追加
            nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.convs(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out




class ResNetBottleneck(nn.Module):
    def __init__(self, depth, n_class=10):
        super().__init__()
        # 指定した深さ（畳み込みの層数）でネットワークを構築できるかを確認
        assert (depth - 2) % 9 == 0, 'When use Bottleneck, depth should be 9n+2 (e.g. 47, 56, 110, 1199).'
        if n_class == 10 or n_class == 100:
            n_blocks = (depth - 2) // 9  # 1ブロックあたりのBasic Blockの数を決定
            
        elif n_class == 200 or n_class == 1000 or n_class == 101:
            n_blocks = 3                  #Tiny ImageNet，ImageNetは各LayerによってBasec Blockの数が異なるので，あらかじめ3としておき，各self._make_layerで＋1などして調節する

        self.n_class = n_class
        
        if n_class == 10:
            self.inplanes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)

            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(256 * Bottleneck.expansion, n_class, bias=False) 
            
        if n_class == 100:
            self.inplanes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)

            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(256 * Bottleneck.expansion, n_class, bias=False) 
            
        if n_class == 200:
            self.inplanes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks+1, stride=2) #公式の実装と同じBlockの数となるように変更 n_blocks = 4
            self.layer3 = self._make_layer(256, n_blocks+3, stride=2) #公式の実装と同じBlockの数となるように変更 n_blocks = 6
            self.layer4 = self._make_layer(512, n_blocks, stride=2)  

            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(512 * Bottleneck.expansion, n_class, bias=False) 
            
        if n_class == 101:
            self.inplanes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks+1, stride=2) #公式の実装と同じBlockの数となるように変更 n_blocks = 4
            self.layer3 = self._make_layer(256, n_blocks+3, stride=2) #公式の実装と同じBlockの数となるように変更 n_blocks = 6
            self.layer4 = self._make_layer(512, n_blocks, stride=2)  

            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(512 * Bottleneck.expansion, n_class, bias=False) 
            
            
        if n_class == 1000:
            self.inplanes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks+1, stride=2) #公式の実装と同じBlockの数となるように変更
            self.layer3 = self._make_layer(256, n_blocks+3, stride=2) #公式の実装と同じBlockの数となるように変更
            self.layer4 = self._make_layer(512, n_blocks, stride=2)  

            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(512 * Bottleneck.expansion, n_class, bias=False)

            
    def _make_layer(self, planes, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(0, n_blocks - 1):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.n_class == 10 or self.n_class == 100:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)  
            x = self.layer2(x)
            x = self.layer3(x)
            #print("x shape",x.shape) #torch.Size([100(batchsize), 1024(conv channel), 8, 8])

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)   #x.size(0)は行数  fc層へ入力するための処理　

            y = self.fc(x)
            
        elif self.n_class == 200 or self.n_class == 1000 or self.n_class == 101:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)  
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)    

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)   #x.size(0)は行数  fc層へ入力するための処理　

            y = self.fc(x)
        
        return x, F.log_softmax(y, dim=1)
    
    
    
class ResNetBottleneck_oneoutput(nn.Module):
    def __init__(self, depth, n_class=10):
        super().__init__()
        # 指定した深さ（畳み込みの層数）でネットワークを構築できるかを確認
        assert (depth - 2) % 9 == 0, 'When use Bottleneck, depth should be 9n+2 (e.g. 47, 56, 110, 1199).'
        if n_class == 10 or n_class == 100:
            n_blocks = (depth - 2) // 9  # 1ブロックあたりのBasic Blockの数を決定
            
        elif n_class == 200 or n_class == 1000 or n_class == 101:
            n_blocks = 3                  #Tiny ImageNet，ImageNetは各LayerによってBasec Blockの数が異なるので，あらかじめ3としておき，各self._make_layerで＋1などして調節する

        self.n_class = n_class
        
        if n_class == 10:
            self.inplanes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)

            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(256 * Bottleneck.expansion, n_class, bias=False) 
            
        if n_class == 100:
            self.inplanes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)

            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(256 * Bottleneck.expansion, n_class, bias=False) 
            
        if n_class == 200:
            self.inplanes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks+1, stride=2) #公式の実装と同じBlockの数となるように変更 n_blocks = 4
            self.layer3 = self._make_layer(256, n_blocks+3, stride=2) #公式の実装と同じBlockの数となるように変更 n_blocks = 6
            self.layer4 = self._make_layer(512, n_blocks, stride=2)  

            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(512 * Bottleneck.expansion, n_class, bias=False) 
            
        if n_class == 101:
            self.inplanes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks+1, stride=2) #公式の実装と同じBlockの数となるように変更 n_blocks = 4
            self.layer3 = self._make_layer(256, n_blocks+3, stride=2) #公式の実装と同じBlockの数となるように変更 n_blocks = 6
            self.layer4 = self._make_layer(512, n_blocks, stride=2)  

            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(512 * Bottleneck.expansion, n_class, bias=False) 
            
        if n_class == 1000:
            self.inplanes = 64
    
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks+1, stride=2) #公式の実装と同じBlockの数となるように変更
            self.layer3 = self._make_layer(256, n_blocks+3, stride=2) #公式の実装と同じBlockの数となるように変更
            self.layer4 = self._make_layer(512, n_blocks, stride=2)  

            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(512 * Bottleneck.expansion, n_class, bias=False) 

            
    def _make_layer(self, planes, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(0, n_blocks - 1):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.n_class == 10 or self.n_class == 100:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)  
            x = self.layer2(x)
            x = self.layer3(x)
            #print("x shape",x.shape) #torch.Size([100(batchsize), 1024(conv channel), 8, 8])

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)   #x.size(0)は行数  fc層へ入力するための処理　

            y = self.fc(x)
            
        elif self.n_class == 200 or self.n_class == 1000 or self.n_class == 101:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)  
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)    

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)   #x.size(0)は行数  fc層へ入力するための処理　

            y = self.fc(x)
        
        return y


##############################################################################################################################










    
    
    
    






##########################Basic ResNet#########################################################################

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True), 
            nn.Dropout(0.3),   #追加
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.convs(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual                 #スキップ結合
        out = self.relu(out)
        return out

class ResNetBasicBlock(nn.Module):
    def __init__(self, depth, n_class):
        super().__init__()
        # 指定した深さ（畳み込みの層数）でネットワークを構築できるかを確認
        assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2 (e.g. 20, 32, 44).'
        if n_class == 10 or n_class == 100:
            n_blocks = (depth - 2) // 6  # 1ブロックあたりのBasic Blockの数を決定
            
        elif n_class == 200 or n_class == 1000 or n_class == 101:
            n_blocks = 2                 #Tiny ImageNet，ImageNetはResNet-18なので2

        self.n_class = n_class
        
        if n_class == 10:
            self.inplanes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)
        
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(256 * BasicBlock.expansion, n_class, bias=False) 
            
        if n_class == 100: #inplanes=16だと精度が0.6848となり10％低下したので64に
            self.inplanes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)
        
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(256 * BasicBlock.expansion, n_class, bias=False) 
            
        if n_class == 200:
            self.inplanes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)
            self.layer4 = self._make_layer(512, n_blocks, stride=2)   
        
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(512 * BasicBlock.expansion, n_class, bias=False)
            
        if n_class == 101:
            self.inplanes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)
            self.layer4 = self._make_layer(512, n_blocks, stride=2)   
        
            self.avgpool = nn.AvgPool2d(16)
            self.fc = nn.Linear(512 * BasicBlock.expansion, n_class, bias=False)
            
        if n_class == 1000:
            self.inplanes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)
            self.layer4 = self._make_layer(512, n_blocks, stride=2)   
        
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(512 * BasicBlock.expansion, n_class, bias=False) 




    def _make_layer(self, planes, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(0, n_blocks - 1):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        if self.n_class == 10 or self.n_class == 100:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)  
            x = self.layer2(x)
            x = self.layer3(x)
            #print("x shape",x.shape) #torch.Size([100(batchsize), 256(conv channel), 8, 8])

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)   #x.size(0)は行数  fc層へ入力するための処理　

            y = self.fc(x)
            
        elif self.n_class == 200 or self.n_class == 1000 or self.n_class == 101:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)  
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)    

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)   #x.size(0)は行数  fc層へ入力するための処理　

            y = self.fc(x)

        
        return x, F.log_softmax(y, dim=1)


    
        
    
# class ResNetBasicBlock_loss2place(nn.Module):
#     def __init__(self, depth, n_class):
#         super().__init__()
#         # 指定した深さ（畳み込みの層数）でネットワークを構築できるかを確認
#         assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2 (e.g. 20, 32, 44).'
#         if n_class == 10 or n_class == 100:
#             n_blocks = (depth - 2) // 6  # 1ブロックあたりのBasic Blockの数を決定
            
#         elif n_class == 200 or n_class == 1000 or n_class == 101:
#             n_blocks = 2                 #Tiny ImageNet，ImageNetはResNet-18なので2

#         self.n_class = n_class
        
#         if n_class == 10 or n_class == 100:
#             self.inplanes = 64

#             self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(64)
#             self.relu = nn.ReLU(inplace=True)

#             self.layer1 = self._make_layer(64, n_blocks)
#             self.layer2 = self._make_layer(128, n_blocks, stride=2)
#             self.avgpool2 = nn.AvgPool2d(16)
#             self.layer3 = self._make_layer(256, n_blocks, stride=2)
                    
#             self.avgpool = nn.AvgPool2d(8)
#             self.fc = nn.Linear(256 * BasicBlock.expansion, n_class, bias=False)

#         if n_class == 200:
#             self.inplanes = 64

#             self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(64)
#             self.relu = nn.ReLU(inplace=True)

#             self.layer1 = self._make_layer(64, n_blocks)
#             self.layer2 = self._make_layer(128, n_blocks, stride=2)
#             self.layer3 = self._make_layer(256, n_blocks, stride=2)
#             self.avgpool2 = nn.AvgPool2d(16)
#             self.layer4 = self._make_layer(512, n_blocks, stride=2)   
        
#             self.avgpool = nn.AvgPool2d(8)
#             self.fc = nn.Linear(512 * BasicBlock.expansion, n_class, bias=False)         
            
#         if n_class == 101:
#             self.inplanes = 64

#             self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(64)
#             self.relu = nn.ReLU(inplace=True)

#             self.layer1 = self._make_layer(64, n_blocks)
#             self.layer2 = self._make_layer(128, n_blocks, stride=2)
#             self.layer3 = self._make_layer(256, n_blocks, stride=2)
#             self.layer4 = self._make_layer(512, n_blocks, stride=2)   
        
#             self.avgpool = nn.AvgPool2d(8)
#             self.fc = nn.Linear(512 * BasicBlock.expansion, n_class, bias=False)
            
#         if n_class == 1000:
#             self.inplanes = 64

#             self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#             self.bn1 = nn.BatchNorm2d(64)
#             self.relu = nn.ReLU(inplace=True)

#             self.layer1 = self._make_layer(64, n_blocks)
#             self.layer2 = self._make_layer(128, n_blocks, stride=2)
#             self.layer3 = self._make_layer(256, n_blocks, stride=2)
#             self.avgpool2 = nn.AvgPool2d(16)
#             self.layer4 = self._make_layer(512, n_blocks, stride=2)   
        
#             self.avgpool = nn.AvgPool2d(8)
#             self.fc = nn.Linear(512 * BasicBlock.expansion, n_class, bias=False)         



#     def _make_layer(self, planes, n_blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * BasicBlock.expansion),
#             )

#         layers = []
#         layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * BasicBlock.expansion
#         for _ in range(0, n_blocks - 1):
#             layers.append(BasicBlock(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         if self.n_class == 10 or self.n_class == 100:
#             x = self.conv1(x)
#             x = self.bn1(x)
#             x = self.relu(x)

#             x = self.layer1(x)  #スキップする間のブロックのこと
#             x = self.layer2(x)
            
#             m =  self.avgpool2(x) 
#             m = m.view(m.size(0), -1) # 256 dimensional
            
#             x = self.layer3(x)

#             x = self.avgpool(x)
#             x = x.view(x.size(0), -1)   #x.size(0)は行数  fc層へ入力するための処理　
        
#             y = self.fc(x)
            
#         elif self.n_class == 200 or self.n_class == 1000 or self.n_class == 101:
#             x = self.conv1(x)
#             x = self.bn1(x)
#             x = self.relu(x)

#             x = self.layer1(x)  #スキップする間のブロックのこと
#             x = self.layer2(x)
#             x = self.layer3(x)
            
#             m =  self.avgpool2(x) 
#             m = m.view(m.size(0), -1) # 256 dimensional
            
#             x = self.layer4(x)    

#             x = self.avgpool(x)
#             x = x.view(x.size(0), -1)   #x.size(0)は行数  fc層へ入力するための処理　
        
#             y = self.fc(x)

        
#         return m, x, F.log_softmax(y, dim=1)



    
class ResNetBasicBlock_oneoutput(nn.Module):
    def __init__(self, depth, n_class):
        super().__init__()
        # 指定した深さ（畳み込みの層数）でネットワークを構築できるかを確認
        assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2 (e.g. 20, 32, 44).'
        if n_class == 10 or n_class == 100:
            n_blocks = (depth - 2) // 6  # 1ブロックあたりのBasic Blockの数を決定
            
        elif n_class == 200 or n_class == 1000 or n_class == 101:
            n_blocks = 2                 #Tiny ImageNet，ImageNetはResNet-18なので2

        self.n_class = n_class
        
        if n_class == 10:
            self.inplanes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)
        
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(256 * BasicBlock.expansion, n_class, bias=False) 
            
        if n_class == 100: #inplanes=16だと精度が0.6848となり10％低下したので64に
            self.inplanes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)
        
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(256 * BasicBlock.expansion, n_class, bias=False) 
            
        if n_class == 200:
            self.inplanes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)
            self.layer4 = self._make_layer(512, n_blocks, stride=2)   
        
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(512 * BasicBlock.expansion, n_class, bias=False) 
            
        if n_class == 101:
            self.inplanes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)
            self.layer4 = self._make_layer(512, n_blocks, stride=2)   
        
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(512 * BasicBlock.expansion, n_class, bias=False)
            
        if n_class == 1000:
            self.inplanes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(64, n_blocks)
            self.layer2 = self._make_layer(128, n_blocks, stride=2)
            self.layer3 = self._make_layer(256, n_blocks, stride=2)
            self.layer4 = self._make_layer(512, n_blocks, stride=2)   
        
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(512 * BasicBlock.expansion, n_class, bias=False) 




    def _make_layer(self, planes, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(0, n_blocks - 1):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        if self.n_class == 10 or self.n_class == 100:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)  
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)   #x.size(0)は行数  fc層へ入力するための処理　

            y = self.fc(x)
            
        elif self.n_class == 200 or self.n_class == 1000 or self.n_class == 101:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)  
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)    

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)   #x.size(0)は行数  fc層へ入力するための処理　

            y = self.fc(x)

        
        return y

    
    
    
# class VGG(nn.Module):
#     def __init__(
#         self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
#     ) -> None:
#         super().__init__()
#         _log_api_usage_once(self)
#         self.features = features
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(4096, num_classes),
#         )
#         if init_weights:
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                     if m.bias is not None:
#                         nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.BatchNorm2d):
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.Linear):
#                     nn.init.normal_(m.weight, 0, 0.01)
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         y = self.classifier(x)
#         return x, F.log_softmax(y, dim=1)


# def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
#     layers: List[nn.Module] = []
#     in_channels = 3
#     for v in cfg:
#         if v == "M":
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             v = cast(int, v)
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)


# cfgs: Dict[str, List[Union[str, int]]] = {
#     "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
#     "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
# }


# def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
#     if weights is not None:
#         kwargs["init_weights"] = False
#         if weights.meta["categories"] is not None:
#             _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
#     model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress))
#     return model


# _COMMON_META = {
#     "min_size": (32, 32),
#     "categories": _IMAGENET_CATEGORIES,
#     "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
#     "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
# }


