import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tool import manipulate


class vgg19Cifar10(nn.Module):
    def __init__(self):
        super(vgg19Cifar10,self).__init__()

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
        self.fc3 = nn.Linear(in_features=100, out_features=10)
        self.bn_f1 = nn.BatchNorm1d(200)
        self.bn_f2 = nn.BatchNorm1d(100)

    def forward(self,x):

        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn2_1(self.conv2_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.relu(self.bn3_1(self.conv3_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = F.relu(self.bn3_4(self.conv3_4(x)))
        x = F.relu(self.bn4_1(self.conv4_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = F.relu(self.bn4_4(self.conv4_4(x)))        
        x = F.relu(self.bn5_1(self.conv5_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        #print(x.shape)
        #quit(0)
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        x = F.relu(self.bn5_4(self.conv5_4(x))) 
        x = F.relu(self.fc1(F.max_pool2d(x, 2).view(-1, 512)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class vgg19ManiCifar10(nn.Module):
    def __init__(self):
        super(vgg19ManiCifar10,self).__init__()

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
        self.fc3 = nn.Linear(in_features=100, out_features=10)
        self.bn_f1 = nn.BatchNorm1d(200)
        self.bn_f2 = nn.BatchNorm1d(100)

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


#########################################
################ CIFAR 100 ##############
#########################################
class vgg19Cifar100(nn.Module):
    def __init__(self):
        super(vgg19Cifar100,self).__init__()

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
        self.bn_f1 = nn.BatchNorm1d(200)
        self.bn_f2 = nn.BatchNorm1d(100)

    def forward(self,x):

        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn2_1(self.conv2_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.relu(self.bn3_1(self.conv3_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = F.relu(self.bn3_4(self.conv3_4(x)))
        x = F.relu(self.bn4_1(self.conv4_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = F.relu(self.bn4_4(self.conv4_4(x)))        
        x = F.relu(self.bn5_1(self.conv5_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        #print(x.shape)
        #quit(0)
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        x = F.relu(self.bn5_4(self.conv5_4(x))) 
        x = F.relu(self.fc1(F.max_pool2d(x, 2).view(-1, 512)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class vgg19ManiCifar100(nn.Module):
    def __init__(self):
        super(vgg19ManiCifar100,self).__init__()

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
        self.bn_f1 = nn.BatchNorm1d(200)
        self.bn_f2 = nn.BatchNorm1d(100)

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


#########################################
################ SVHN ###################
#########################################
vgg19SVHN = vgg19Cifar10
vgg19ManiSVHN = vgg19ManiCifar10