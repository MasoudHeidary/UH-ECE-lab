import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tool import manipulate



class squeezenetCifar10(nn.Module):
    def __init__(self):
        super(squeezenetCifar10,self).__init__()

        # 1
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)

        # 2
        self.conv2_1 = nn.Conv2d(in_channels=96,out_channels=16,kernel_size=1)
        self.conv2_2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=1)
        self.conv2_3 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn2_3 = nn.BatchNorm2d(64)
        
        # 3
        self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=16,kernel_size=1)
        self.conv3_2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=1)
        self.conv3_3 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(16)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.bn3_3 = nn.BatchNorm2d(64)

        # 4
        self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=32,kernel_size=1)
        self.conv4_2 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=1)
        self.conv4_3 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(32)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.bn4_3 = nn.BatchNorm2d(128)
        
        # 5
        self.conv5_1 = nn.Conv2d(in_channels=256,out_channels=32,kernel_size=1)
        self.conv5_2 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=1)
        self.conv5_3 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(32)
        self.bn5_2 = nn.BatchNorm2d(128)
        self.bn5_3 = nn.BatchNorm2d(128)

        # 6
        self.conv6_1 = nn.Conv2d(in_channels=256,out_channels=48,kernel_size=1)
        self.conv6_2 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=1)
        self.conv6_3 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(48)
        self.bn6_2 = nn.BatchNorm2d(192)
        self.bn6_3 = nn.BatchNorm2d(192)

        # 7
        self.conv7_1 = nn.Conv2d(in_channels=384,out_channels=48,kernel_size=1)
        self.conv7_2 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=1)
        self.conv7_3 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(48)
        self.bn7_2 = nn.BatchNorm2d(192)
        self.bn7_3 = nn.BatchNorm2d(192)

        # 8
        self.conv8_1 = nn.Conv2d(in_channels=384,out_channels=64,kernel_size=1)
        self.conv8_2 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1)
        self.conv8_3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3, padding=1)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.bn8_2 = nn.BatchNorm2d(256)
        self.bn8_3 = nn.BatchNorm2d(256)

        # 9
        self.conv9_1 = nn.Conv2d(in_channels=512,out_channels=64,kernel_size=1)
        self.conv9_2 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1)
        self.conv9_3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3, padding=1)
        self.bn9_1 = nn.BatchNorm2d(64)
        self.bn9_2 = nn.BatchNorm2d(256)
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


class squeezenetManiCifar10(nn.Module):
    def __init__(self):
        super(squeezenetManiCifar10,self).__init__()

        # 1
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)

        # 2
        self.conv2_1 = nn.Conv2d(in_channels=96,out_channels=16,kernel_size=1)
        self.conv2_2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=1)
        self.conv2_3 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn2_3 = nn.BatchNorm2d(64)
        
        # 3
        self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=16,kernel_size=1)
        self.conv3_2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=1)
        self.conv3_3 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(16)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.bn3_3 = nn.BatchNorm2d(64)

        # 4
        self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=32,kernel_size=1)
        self.conv4_2 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=1)
        self.conv4_3 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(32)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.bn4_3 = nn.BatchNorm2d(128)
        
        # 5
        self.conv5_1 = nn.Conv2d(in_channels=256,out_channels=32,kernel_size=1)
        self.conv5_2 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=1)
        self.conv5_3 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(32)
        self.bn5_2 = nn.BatchNorm2d(128)
        self.bn5_3 = nn.BatchNorm2d(128)

        # 6
        self.conv6_1 = nn.Conv2d(in_channels=256,out_channels=48,kernel_size=1)
        self.conv6_2 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=1)
        self.conv6_3 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(48)
        self.bn6_2 = nn.BatchNorm2d(192)
        self.bn6_3 = nn.BatchNorm2d(192)

        # 7
        self.conv7_1 = nn.Conv2d(in_channels=384,out_channels=48,kernel_size=1)
        self.conv7_2 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=1)
        self.conv7_3 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(48)
        self.bn7_2 = nn.BatchNorm2d(192)
        self.bn7_3 = nn.BatchNorm2d(192)

        # 8
        self.conv8_1 = nn.Conv2d(in_channels=384,out_channels=64,kernel_size=1)
        self.conv8_2 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1)
        self.conv8_3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3, padding=1)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.bn8_2 = nn.BatchNorm2d(256)
        self.bn8_3 = nn.BatchNorm2d(256)

        # 9
        self.conv9_1 = nn.Conv2d(in_channels=512,out_channels=64,kernel_size=1)
        self.conv9_2 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1)
        self.conv9_3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3, padding=1)
        self.bn9_1 = nn.BatchNorm2d(64)
        self.bn9_2 = nn.BatchNorm2d(256)
        self.bn9_3 = nn.BatchNorm2d(256)

        #self.conv10 = nn.Conv2d(in_channels=512,out_channels=10,kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=512, out_features=10) # adding this to simplify the output shape

    def forward(self,x):
        x = F.relu(self.bn1(manipulate(self.conv1(x))))
        x = F.max_pool2d(x,2)

        x = F.relu(self.bn2_1(manipulate(self.conv2_1(x))))
        x1 = F.relu(self.bn2_2(manipulate(self.conv2_2(x))))
        x2 = F.relu(self.bn2_3(manipulate(self.conv2_3(x))))
        x = torch.cat((x1,x2),dim = 1)

        x = F.relu(self.bn3_1(manipulate(self.conv3_1(x))))
        x1 = F.relu(self.bn3_2(manipulate(self.conv3_2(x))))
        x2 = F.relu(self.bn3_3(manipulate(self.conv3_3(x))))
        x = torch.cat((x1,x2),dim = 1)

        x = F.relu(self.bn4_1(manipulate(self.conv4_1(x))))
        x1 = F.relu(self.bn4_2(manipulate(self.conv4_2(x))))
        x2 = F.relu(self.bn4_3(manipulate(self.conv4_3(x))))
        x = torch.cat((x1,x2),dim = 1)
        x = F.max_pool2d(x,2)

        x = F.relu(self.bn5_1(manipulate(self.conv5_1(x))))
        x1 = F.relu(self.bn5_2(manipulate(self.conv5_2(x))))
        x2 = F.relu(self.bn5_3(manipulate(self.conv5_3(x))))
        x = torch.cat((x1,x2),dim = 1)

        x = F.relu(self.bn6_1(manipulate(self.conv6_1(x))))
        x1 = F.relu(self.bn6_2(manipulate(self.conv6_2(x))))
        x2 = F.relu(self.bn6_3(manipulate(self.conv6_3(x))))
        x = torch.cat((x1,x2),dim = 1)

        x = F.relu(self.bn7_1(manipulate(self.conv7_1(x))))
        x1 = F.relu(self.bn7_2(manipulate(self.conv7_2(x))))
        x2 = F.relu(self.bn7_3(manipulate(self.conv7_3(x))))
        x = torch.cat((x1,x2),dim = 1)

        x = F.relu(self.bn8_1(manipulate(self.conv8_1(x))))
        x1 = F.relu(self.bn8_2(manipulate(self.conv8_2(x))))
        x2 = F.relu(self.bn8_3(manipulate(self.conv8_3(x))))
        x = torch.cat((x1,x2),dim = 1)
        x = F.max_pool2d(x,2)

        x = F.relu(self.bn9_1(manipulate(self.conv9_1(x))))
        x1 = F.relu(self.bn9_2(manipulate(self.conv9_2(x))))
        x2 = F.relu(self.bn9_3(manipulate(self.conv9_3(x))))
        x = torch.cat((x1,x2),dim = 1)

        #x = F.relu((self.conv10(x)))
        x = self.avgpool(x).view(-1, 512)
        x = manipulate(self.fc(x))

        return F.log_softmax(x, dim=1)

