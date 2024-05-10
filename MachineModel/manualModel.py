import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class darknet(nn.Module):
    def __init__(self):
        super(darknet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1, padding=0)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.conv3_3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(128)
        
        self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1, padding=0)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.conv4_3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(256)
        
        self.conv5_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1, padding=0)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.conv5_3 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1, padding=1)
        self.bn5_4 = nn.BatchNorm2d(256)
        self.conv5_5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.bn5_5 = nn.BatchNorm2d(512)

        self.conv6_1 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(1024)
        self.conv6_2 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1, padding=0)
        self.bn6_2 = nn.BatchNorm2d(512)
        self.conv6_3 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, padding=1)
        self.bn6_3 = nn.BatchNorm2d(1024)
        self.conv6_4 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1, padding=1)
        self.bn6_4 = nn.BatchNorm2d(512)
        self.conv6_5 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, padding=1)
        self.bn6_5 = nn.BatchNorm2d(1024)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=1024, out_features=10)

    def forward(self,x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        x = F.relu(self.bn5_4(self.conv5_4(x)))
        x = F.relu(self.bn5_5(self.conv5_5(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn6_1(self.conv6_1(x)))
        x = F.relu(self.bn6_2(self.conv6_2(x)))
        x = F.relu(self.bn6_3(self.conv6_3(x)))
        x = F.relu(self.bn6_4(self.conv6_4(x)))
        x = F.relu(self.bn6_5(self.conv6_5(x)))

        x = self.avgpool(x).view(-1, 1024)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1)
        self.bn2_3 = nn.BatchNorm2d(64)
        self.conv2_4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1)
        self.bn2_4 = nn.BatchNorm2d(64)
        
        self.conv3_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.conv3_4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(128)
        
        self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.conv4_3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(256)
        self.conv4_4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(256)
        
        self.conv5_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(in_features=512, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)
        
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))  

        x1 = F.relu(self.bn2_1(self.conv2_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn2_2(self.conv2_2(x1)))
        x = torch.add(x1, x)

        x1 = F.relu(self.bn2_3(self.conv2_3(x)))
        x = F.relu(self.bn2_4(self.conv2_4(x1)))
        x=torch.add(x1, x)

        x1 = F.relu(self.bn3_1(self.conv3_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn3_2(self.conv3_2(x1)))
        x = torch.add(x1, x)

        x1 = F.relu(self.bn3_3(self.conv3_3(x)))
        x = F.relu(self.bn3_4(self.conv3_4(x1)))
        x = torch.add(x1, x)

        x1 = F.relu(self.bn4_1(self.conv4_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn4_2(self.conv4_2(x1)))
        x = torch.add(x1, x)

        x1 = F.relu(self.bn4_3(self.conv4_3(x)))
        x = F.relu(self.bn4_4(self.conv4_4(x1)))
        x = torch.add(x1, x)

        x1 = F.relu(self.bn5_1(self.conv5_1(F.max_pool2d(x, 2))))
        x = F.relu(self.bn5_2(self.conv5_2(x1)))
        x = torch.add(x1, x)

        x1 = F.relu(self.bn5_3(self.conv5_3(x)))
        x = F.relu(self.bn5_4(self.conv5_4(x1)))
        x = torch.add(x1, x)
        
        x = F.relu((self.fc1(F.avg_pool2d(x,2).view(-1, 512))))
        x = F.relu(self.fc2(x))
        
        return F.log_softmax(x, dim=1)

class squeezenet(nn.Module):
    def __init__(self):
        super(squeezenet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2_1 = nn.Conv2d(in_channels=96,out_channels=16,kernel_size=1)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.conv2_2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3, padding=1)
        self.bn2_3 = nn.BatchNorm2d(64)
        
        self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=16,kernel_size=1)
        self.bn3_1 = nn.BatchNorm2d(16)
        self.conv3_2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=1)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.conv3_3 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(64)

        self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=32,kernel_size=1)
        self.bn4_1 = nn.BatchNorm2d(32)
        self.conv4_2 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=1)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.conv4_3 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(128)
        
        self.conv5_1 = nn.Conv2d(in_channels=256,out_channels=32,kernel_size=1)
        self.bn5_1 = nn.BatchNorm2d(32)
        self.conv5_2 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=1)
        self.bn5_2 = nn.BatchNorm2d(128)
        self.conv5_3 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(128)

        self.conv6_1 = nn.Conv2d(in_channels=256,out_channels=48,kernel_size=1)
        self.bn6_1 = nn.BatchNorm2d(48)
        self.conv6_2 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=1)
        self.bn6_2 = nn.BatchNorm2d(192)
        self.conv6_3 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=3, padding=1)
        self.bn6_3 = nn.BatchNorm2d(192)

        self.conv7_1 = nn.Conv2d(in_channels=384,out_channels=48,kernel_size=1)
        self.bn7_1 = nn.BatchNorm2d(48)
        self.conv7_2 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=1)
        self.bn7_2 = nn.BatchNorm2d(192)
        self.conv7_3 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=3, padding=1)
        self.bn7_3 = nn.BatchNorm2d(192)

        self.conv8_1 = nn.Conv2d(in_channels=384,out_channels=64,kernel_size=1)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1)
        self.bn8_2 = nn.BatchNorm2d(256)
        self.conv8_3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3, padding=1)
        self.bn8_3 = nn.BatchNorm2d(256)

        self.conv9_1 = nn.Conv2d(in_channels=512,out_channels=64,kernel_size=1)
        self.bn9_1 = nn.BatchNorm2d(64)
        self.conv9_2 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1)
        self.bn9_2 = nn.BatchNorm2d(256)
        self.conv9_3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3, padding=1)
        self.bn9_3 = nn.BatchNorm2d(256)

        #self.conv10 = nn.Conv2d(in_channels=512,out_channels=10,kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=512, out_features=10) # adding this to simplify the output shape

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x,2)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x1 = F.relu(self.bn2_2(self.conv2_2(x)))
        x2 = F.relu(self.bn2_3(self.conv2_3(x)))
        x = torch.cat((x1,x2),dim = 1)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x1 = F.relu(self.bn3_2(self.conv3_2(x)))
        x2 = F.relu(self.bn3_3(self.conv3_3(x)))
        x = torch.cat((x1,x2),dim = 1)

        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x1 = F.relu(self.bn4_2(self.conv4_2(x)))
        x2 = F.relu(self.bn4_3(self.conv4_3(x)))
        x = torch.cat((x1,x2),dim = 1)
        x = F.max_pool2d(x,2)

        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x1 = F.relu(self.bn5_2(self.conv5_2(x)))
        x2 = F.relu(self.bn5_3(self.conv5_3(x)))
        x = torch.cat((x1,x2),dim = 1)

        x = F.relu(self.bn6_1(self.conv6_1(x)))
        x1 = F.relu(self.bn6_2(self.conv6_2(x)))
        x2 = F.relu(self.bn6_3(self.conv6_3(x)))
        x = torch.cat((x1,x2),dim = 1)

        x = F.relu(self.bn7_1(self.conv7_1(x)))
        x1 = F.relu(self.bn7_2(self.conv7_2(x)))
        x2 = F.relu(self.bn7_3(self.conv7_3(x)))
        x = torch.cat((x1,x2),dim = 1)

        x = F.relu(self.bn8_1(self.conv8_1(x)))
        x1 = F.relu(self.bn8_2(self.conv8_2(x)))
        x2 = F.relu(self.bn8_3(self.conv8_3(x)))
        x = torch.cat((x1,x2),dim = 1)
        x = F.max_pool2d(x,2)

        x = F.relu(self.bn9_1(self.conv9_1(x)))
        x1 = F.relu(self.bn9_2(self.conv9_2(x)))
        x2 = F.relu(self.bn9_3(self.conv9_3(x)))
        x = torch.cat((x1,x2),dim = 1)

        #x = F.relu((self.conv10(x)))
        x = self.avgpool(x).view(-1, 512)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

class vgg11(nn.Module):
    def __init__(self):
        super(vgg11,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
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

class vgg19(nn.Module):
    def __init__(self):
        super(vgg19,self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256)
        
        self.conv4_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.conv4_4 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(512)
        
        self.conv5_1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(in_features=512, out_features=200)
        self.bn_f1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(in_features=200, out_features=100)
        self.bn_f2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(in_features=100, out_features=10)

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
        x = F.relu((self.fc1(F.max_pool2d(x, 2).view(-1, 512))))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)




