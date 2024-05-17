import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tool import *

#TODO: fix the output size to 10
class vgg11Cifar10(nn.Module):
    def __init__(self):
        super(vgg11Cifar10,self).__init__()
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

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(F.max_pool2d(x, 2))))
        x = F.relu(self.bn3_1(self.conv3_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn4_1(self.conv4_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        #print(x.shape)
        #uit(0)
        x = F.relu(self.bn5_1(self.conv5_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.fc1(F.dropout(F.max_pool2d(x, 2).view(-1, 512))))
        x = F.relu(self.fc2(F.dropout(x)))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class vgg11ManiCifar10(nn.Module):
    def __init__(self):
        super(vgg11ManiCifar10,self).__init__()
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



#########################################
################ CIFAR 100 ##############
#########################################
class vgg11Cifar100(nn.Module):
    def __init__(self):
        super(vgg11Cifar100,self).__init__()
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

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(F.max_pool2d(x, 2))))
        x = F.relu(self.bn3_1(self.conv3_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn4_1(self.conv4_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        #print(x.shape)
        #uit(0)
        x = F.relu(self.bn5_1(self.conv5_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.fc1(F.dropout(F.max_pool2d(x, 2).view(-1, 512))))
        x = F.relu(self.fc2(F.dropout(x)))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class vgg11ManiCifar100(nn.Module):
    def __init__(self):
        super(vgg11ManiCifar100,self).__init__()
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
