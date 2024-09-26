import torch
import torch.nn as nn

from .mani import *
import torch.nn.functional as F


# class darknet(nn.Module):
#     def __init__(self):
#         super(darknet,self).__init__()

#         d_rate = 0.25

#         ### 1
#         self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.dropout1 = nn.Dropout(d_rate)
        
#         ### 2
#         self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.dropout2 = nn.Dropout(d_rate)
        
#         ### 3
#         self.conv3_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
#         self.conv3_2 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1, padding=0)
#         self.conv3_3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
#         self.bn3_1 = nn.BatchNorm2d(128)
#         self.bn3_2 = nn.BatchNorm2d(64)
#         self.bn3_3 = nn.BatchNorm2d(128)
#         self.dropout3 = nn.Dropout(d_rate)

#         ### 4
#         self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
#         self.conv4_2 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1, padding=0)
#         self.conv4_3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
#         self.bn4_1 = nn.BatchNorm2d(256)
#         self.bn4_2 = nn.BatchNorm2d(128)
#         self.bn4_3 = nn.BatchNorm2d(256)
#         self.dropout4 = nn.Dropout(d_rate)
        
#         ### 5
#         self.conv5_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
#         self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1, padding=0)
#         self.conv5_3 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
#         self.conv5_4 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1, padding=1)
#         self.conv5_5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
#         self.bn5_1 = nn.BatchNorm2d(512)
#         self.bn5_2 = nn.BatchNorm2d(256)
#         self.bn5_3 = nn.BatchNorm2d(512)
#         self.bn5_4 = nn.BatchNorm2d(256)
#         self.bn5_5 = nn.BatchNorm2d(512)
#         self.dropout5 = nn.Dropout(d_rate)

#         ### 6
#         self.conv6_1 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, padding=1)
#         self.conv6_2 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1, padding=0)
#         self.conv6_3 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, padding=1)
#         self.conv6_4 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1, padding=1)
#         self.conv6_5 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, padding=1)
#         self.bn6_1 = nn.BatchNorm2d(1024)
#         self.bn6_2 = nn.BatchNorm2d(512)
#         self.bn6_3 = nn.BatchNorm2d(1024)
#         self.bn6_4 = nn.BatchNorm2d(512)
#         self.bn6_5 = nn.BatchNorm2d(1024)
#         self.dropout6 = nn.Dropout(d_rate)

#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
#         self.fc = nn.Linear(in_features=1024, out_features=100)

#     def forward(self,x):

#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.dropout1(x)
#         x = F.max_pool2d(x, 2)

#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.dropout2(x)
#         x = F.max_pool2d(x, 2)

#         x = F.relu(self.bn3_1(self.conv3_1(x)))
#         x = F.relu(self.bn3_2(self.conv3_2(x)))
#         x = F.relu(self.bn3_3(self.conv3_3(x)))
#         x = self.dropout3(x)
#         x = F.max_pool2d(x, 2)

#         x = F.relu(self.bn4_1(self.conv4_1(x)))
#         x = F.relu(self.bn4_2(self.conv4_2(x)))
#         x = F.relu(self.bn4_3(self.conv4_3(x)))
#         x = self.dropout4(x)
#         x = F.max_pool2d(x, 2)

#         x = F.relu(self.bn5_1(self.conv5_1(x)))
#         x = F.relu(self.bn5_2(self.conv5_2(x)))
#         x = F.relu(self.bn5_3(self.conv5_3(x)))
#         x = F.relu(self.bn5_4(self.conv5_4(x)))
#         x = F.relu(self.bn5_5(self.conv5_5(x)))
#         x = self.dropout5(x)
#         x = F.max_pool2d(x, 2)

#         x = F.relu(self.bn6_1(self.conv6_1(x)))
#         x = F.relu(self.bn6_2(self.conv6_2(x)))
#         x = F.relu(self.bn6_3(self.conv6_3(x)))
#         x = F.relu(self.bn6_4(self.conv6_4(x)))
#         x = F.relu(self.bn6_5(self.conv6_5(x)))
#         x = self.dropout6(x)

#         x = self.avgpool(x).view(-1, 1024)
#         x = self.fc(x)

#         return F.log_softmax(x, dim=1)


class darknet(nn.Module):
    def __init__(self):
        super(darknet,self).__init__()

        ### 1
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        ### 2
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        ### 3
        self.conv3_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1, padding=0)
        self.conv3_3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.bn3_3 = nn.BatchNorm2d(128)

        ### 4
        self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1, padding=0)
        self.conv4_3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.bn4_3 = nn.BatchNorm2d(256)
        
        ### 5
        self.conv5_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1, padding=0)
        self.conv5_3 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1, padding=1)
        self.conv5_5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.bn5_4 = nn.BatchNorm2d(256)
        self.bn5_5 = nn.BatchNorm2d(512)

        ### 6
        self.conv6_1 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1, padding=0)
        self.conv6_3 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, padding=1)
        self.conv6_4 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1, padding=1)
        self.conv6_5 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(1024)
        self.bn6_2 = nn.BatchNorm2d(512)
        self.bn6_3 = nn.BatchNorm2d(1024)
        self.bn6_4 = nn.BatchNorm2d(512)
        self.bn6_5 = nn.BatchNorm2d(1024)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=1024, out_features=100)

    def forward(self,x):

        x = F.relu(self.bn1(manipulate(self.conv1(x))))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(manipulate(self.conv2(x))))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3_1(manipulate(self.conv3_1(x))))
        x = F.relu(self.bn3_2(manipulate(self.conv3_2(x))))
        x = F.relu(self.bn3_3(manipulate(self.conv3_3(x))))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn4_1(manipulate(self.conv4_1(x))))
        x = F.relu(self.bn4_2(manipulate(self.conv4_2(x))))
        x = F.relu(self.bn4_3(manipulate(self.conv4_3(x))))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn5_1(manipulate(self.conv5_1(x))))
        x = F.relu(self.bn5_2(manipulate(self.conv5_2(x))))
        x = F.relu(self.bn5_3(manipulate(self.conv5_3(x))))
        x = F.relu(self.bn5_4(manipulate(self.conv5_4(x))))
        x = F.relu(self.bn5_5(manipulate(self.conv5_5(x))))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn6_1(manipulate(self.conv6_1(x))))
        x = F.relu(self.bn6_2(manipulate(self.conv6_2(x))))
        x = F.relu(self.bn6_3(manipulate(self.conv6_3(x))))
        x = F.relu(self.bn6_4(manipulate(self.conv6_4(x))))
        x = F.relu(self.bn6_5(manipulate(self.conv6_5(x))))

        x = self.avgpool(x).view(-1, 1024)
        x = manipulate(self.fc(x))

        return F.log_softmax(x, dim=1)
