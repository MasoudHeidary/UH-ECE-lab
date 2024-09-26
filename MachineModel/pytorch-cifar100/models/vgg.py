"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

from .mani import *

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


import torch.nn.functional as F

# def vgg11_bn():
#     return VGG(make_layers(cfg['A'], batch_norm=True))

# class vgg11_bn(nn.Module):
#     def __init__(self):
#         super(vgg11_bn,self).__init__()
#         # 1
#         self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)

#         # 2
#         self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)

#         # 3
#         self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
#         self.conv3_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
#         self.bn3_1 = nn.BatchNorm2d(256)
#         self.bn3_2 = nn.BatchNorm2d(256)

#         # 4
#         self.conv4_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
#         self.conv4_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
#         self.bn4_1 = nn.BatchNorm2d(512)
#         self.bn4_2 = nn.BatchNorm2d(512)

#         # 5
#         self.conv5_1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
#         self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
#         self.bn5_1 = nn.BatchNorm2d(512)
#         self.bn5_2 = nn.BatchNorm2d(512)

#         self.fc1 = nn.Linear(in_features=512, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=512)
#         self.fc3 = nn.Linear(in_features=512, out_features=100)

#     def forward(self,x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(F.max_pool2d(x, 2))))
#         x = F.relu(self.bn3_1(self.conv3_1(F.max_pool2d(x, 2))))
#         x = F.relu(self.bn3_2(self.conv3_2(x)))
#         x = F.relu(self.bn4_1(self.conv4_1(F.max_pool2d(x, 2))))
#         x = F.relu(self.bn4_2(self.conv4_2(x)))
#         #print(x.shape)
#         #uit(0)
#         x = F.relu(self.bn5_1(self.conv5_1(F.max_pool2d(x, 2))))
#         x = F.relu(self.bn5_2(self.conv5_2(x)))
#         x = F.relu(self.fc1(F.dropout(F.max_pool2d(x, 2).view(-1, 512))))
#         x = F.relu(self.fc2(F.dropout(x)))
#         x = self.fc3(x)

#         return F.log_softmax(x, dim=1)


class vgg11_bn(nn.Module):
    def __init__(self):
        super(vgg11_bn,self).__init__()
        # 1
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # 2
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # 3
        self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        # 4
        self.conv4_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        # 5
        self.conv5_1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.bn5_2 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(in_features=512, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=100)

        self.fc1 = nn.Linear(in_features=512, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=100)

    def forward(self,x):
        x = F.relu(self.bn1(manipulate(self.conv1(x))))
        x = F.relu(self.bn2(manipulate(self.conv2(F.max_pool2d(x, 2)))))
        x = F.relu(self.bn3_1(manipulate(self.conv3_1(F.max_pool2d(x, 2)))))
        x = F.relu(self.bn3_2(manipulate(self.conv3_2(x))))
        x = F.relu(self.bn4_1(manipulate(self.conv4_1(F.max_pool2d(x, 2)))))
        x = F.relu(self.bn4_2(manipulate(self.conv4_2(x))))
        #print(x.shape)
        #uit(0)
        x = F.relu(self.bn5_1(manipulate(self.conv5_1(F.max_pool2d(x, 2)))))
        x = F.relu(self.bn5_2(manipulate(self.conv5_2(x))))
        x = F.relu(manipulate(self.fc1(F.dropout(F.max_pool2d(x, 2).view(-1, 512)))))
        x = F.relu(manipulate(self.fc2(F.dropout(x))))
        x = manipulate(self.fc3(x))

        return F.log_softmax(x, dim=1)


def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

# def vgg19_bn():
#     return VGG(make_layers(cfg['E'], batch_norm=True))

# class vgg19_bn(nn.Module):
#     def __init__(self):
#         super(vgg19_bn,self).__init__()

#         # 1
#         self.conv1_1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3, padding=1)
#         self.conv1_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1)
#         self.bn1_1 = nn.BatchNorm2d(64)
#         self.bn1_2 = nn.BatchNorm2d(64)
        
#         # 2
#         self.conv2_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
#         self.conv2_2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, padding=1)
#         self.bn2_1 = nn.BatchNorm2d(128)
#         self.bn2_2 = nn.BatchNorm2d(128)
        
#         # 3
#         self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
#         self.conv3_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
#         self.conv3_3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
#         self.conv3_4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
#         self.bn3_1 = nn.BatchNorm2d(256)
#         self.bn3_2 = nn.BatchNorm2d(256)
#         self.bn3_3 = nn.BatchNorm2d(256)
#         self.bn3_4 = nn.BatchNorm2d(256)
        
#         # 4
#         self.conv4_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
#         self.conv4_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
#         self.conv4_3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
#         self.conv4_4 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
#         self.bn4_1 = nn.BatchNorm2d(512)
#         self.bn4_2 = nn.BatchNorm2d(512)
#         self.bn4_3 = nn.BatchNorm2d(512)
#         self.bn4_4 = nn.BatchNorm2d(512)
        
#         # 5
#         self.conv5_1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
#         self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
#         self.conv5_3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
#         self.conv5_4 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
#         self.bn5_1 = nn.BatchNorm2d(512)
#         self.bn5_2 = nn.BatchNorm2d(512)
#         self.bn5_3 = nn.BatchNorm2d(512)
#         self.bn5_4 = nn.BatchNorm2d(512)

#         self.fc1 = nn.Linear(in_features=512, out_features=200)
#         self.fc2 = nn.Linear(in_features=200, out_features=100)
#         self.fc3 = nn.Linear(in_features=100, out_features=100)
#         # self.bn_f1 = nn.BatchNorm1d(200)
#         # self.bn_f2 = nn.BatchNorm1d(100)

#     def forward(self,x):

#         x = F.relu(self.bn1_1(self.conv1_1(x)))
#         x = F.relu(self.bn1_2(self.conv1_2(x)))
#         x = F.relu(self.bn2_1(self.conv2_1(F.max_pool2d(x, 2))))
#         x = F.relu(self.bn2_2(self.conv2_2(x)))
#         x = F.relu(self.bn3_1(self.conv3_1(F.max_pool2d(x, 2))))
#         x = F.relu(self.bn3_2(self.conv3_2(x)))
#         x = F.relu(self.bn3_3(self.conv3_3(x)))
#         x = F.relu(self.bn3_4(self.conv3_4(x)))
#         x = F.relu(self.bn4_1(self.conv4_1(F.max_pool2d(x, 2))))
#         x = F.relu(self.bn4_2(self.conv4_2(x)))
#         x = F.relu(self.bn4_3(self.conv4_3(x)))
#         x = F.relu(self.bn4_4(self.conv4_4(x)))        
#         x = F.relu(self.bn5_1(self.conv5_1(F.max_pool2d(x, 2))))
#         x = F.relu(self.bn5_2(self.conv5_2(x)))
#         #print(x.shape)
#         #quit(0)
#         x = F.relu(self.bn5_3(self.conv5_3(x)))
#         x = F.relu(self.bn5_4(self.conv5_4(x))) 
#         x = F.relu(self.fc1(F.max_pool2d(x, 2).view(-1, 512)))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)

#         return F.log_softmax(x, dim=1)



class vgg19_bn(nn.Module):
    def __init__(self):
        super(vgg19_bn,self).__init__()

        # 1
        self.conv1_1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        # 2
        self.conv2_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        # 3
        self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.bn3_4 = nn.BatchNorm2d(256)
        
        # 4
        self.conv4_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.bn4_4 = nn.BatchNorm2d(512)
        
        # 5
        self.conv5_1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.bn5_4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(in_features=512, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=100)
        # self.bn_f1 = nn.BatchNorm1d(200)
        # self.bn_f2 = nn.BatchNorm1d(100)

    def forward(self,x):

        x = F.relu(self.bn1_1(manipulate(self.conv1_1(x))))
        x = F.relu(self.bn1_2(manipulate(self.conv1_2(x))))
        x = F.relu(self.bn2_1(manipulate(self.conv2_1(F.max_pool2d(x, 2)))))
        x = F.relu(self.bn2_2(manipulate(self.conv2_2(x))))
        x = F.relu(self.bn3_1(manipulate(self.conv3_1(F.max_pool2d(x, 2)))))
        x = F.relu(self.bn3_2(manipulate(self.conv3_2(x))))
        x = F.relu(self.bn3_3(manipulate(self.conv3_3(x))))
        x = F.relu(self.bn3_4(manipulate(self.conv3_4(x))))
        x = F.relu(self.bn4_1(manipulate(self.conv4_1(F.max_pool2d(x, 2)))))
        x = F.relu(self.bn4_2(manipulate(self.conv4_2(x))))
        x = F.relu(self.bn4_3(manipulate(self.conv4_3(x))))
        x = F.relu(self.bn4_4(manipulate(self.conv4_4(x)))        )
        x = F.relu(self.bn5_1(manipulate(self.conv5_1(F.max_pool2d(x, 2)))))
        x = F.relu(self.bn5_2(manipulate(self.conv5_2(x))))
        #print(x.shape)
        #quit(0)
        x = F.relu(self.bn5_3(manipulate(self.conv5_3(x))))
        x = F.relu(self.bn5_4(manipulate(self.conv5_4(x))) )
        x = F.relu(manipulate(self.fc1(F.max_pool2d(x, 2).view(-1, 512))))
        x = F.relu(manipulate(self.fc2(x)))
        x = manipulate(self.fc3(x))

        return F.log_softmax(x, dim=1)


