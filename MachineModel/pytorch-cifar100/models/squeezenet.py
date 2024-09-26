"""squeezenet in pytorch



[1] Song Han, Jeff Pool, John Tran, William J. Dally

    squeezenet: Learning both Weights and Connections for Efficient Neural Networks
    https://arxiv.org/abs/1506.02626
"""

import torch
import torch.nn as nn

from .mani import *

class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):

        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x

class SqueezeNet(nn.Module):

    """mobile net with simple bypass"""
    def __init__(self, class_num=100):

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(96, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)

        self.conv10 = nn.Conv2d(512, class_num, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.stem(x)

        f2 = self.fire2(x)
        f3 = self.fire3(f2) + f2
        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)

        f5 = self.fire5(f4) + f4
        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6
        f8 = self.fire8(f7)
        f8 = self.maxpool(f8)

        f9 = self.fire9(f8)
        c10 = self.conv10(f9)

        x = self.avg(c10)
        x = x.view(x.size(0), -1)

        return x

# def squeezenet(class_num=100):
#     return SqueezeNet(class_num=class_num)

import torch.nn.functional as F

# class squeezenet(nn.Module):
#     def __init__(self):
#         super(squeezenet,self).__init__()

#         # 1
#         self.conv1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(96)

#         # 2
#         self.conv2_1 = nn.Conv2d(in_channels=96,out_channels=16,kernel_size=1)
#         self.conv2_2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=1)
#         self.conv2_3 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3, padding=1)
#         self.bn2_1 = nn.BatchNorm2d(16)
#         self.bn2_2 = nn.BatchNorm2d(64)
#         self.bn2_3 = nn.BatchNorm2d(64)
        
#         # 3
#         self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=16,kernel_size=1)
#         self.conv3_2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=1)
#         self.conv3_3 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3, padding=1)
#         self.bn3_1 = nn.BatchNorm2d(16)
#         self.bn3_2 = nn.BatchNorm2d(64)
#         self.bn3_3 = nn.BatchNorm2d(64)

#         # 4
#         self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=32,kernel_size=1)
#         self.conv4_2 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=1)
#         self.conv4_3 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=3, padding=1)
#         self.bn4_1 = nn.BatchNorm2d(32)
#         self.bn4_2 = nn.BatchNorm2d(128)
#         self.bn4_3 = nn.BatchNorm2d(128)
        
#         # 5
#         self.conv5_1 = nn.Conv2d(in_channels=256,out_channels=32,kernel_size=1)
#         self.conv5_2 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=1)
#         self.conv5_3 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=3, padding=1)
#         self.bn5_1 = nn.BatchNorm2d(32)
#         self.bn5_2 = nn.BatchNorm2d(128)
#         self.bn5_3 = nn.BatchNorm2d(128)

#         # 6
#         self.conv6_1 = nn.Conv2d(in_channels=256,out_channels=48,kernel_size=1)
#         self.conv6_2 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=1)
#         self.conv6_3 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=3, padding=1)
#         self.bn6_1 = nn.BatchNorm2d(48)
#         self.bn6_2 = nn.BatchNorm2d(192)
#         self.bn6_3 = nn.BatchNorm2d(192)

#         # 7
#         self.conv7_1 = nn.Conv2d(in_channels=384,out_channels=48,kernel_size=1)
#         self.conv7_2 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=1)
#         self.conv7_3 = nn.Conv2d(in_channels=48,out_channels=192,kernel_size=3, padding=1)
#         self.bn7_1 = nn.BatchNorm2d(48)
#         self.bn7_2 = nn.BatchNorm2d(192)
#         self.bn7_3 = nn.BatchNorm2d(192)

#         # 8
#         self.conv8_1 = nn.Conv2d(in_channels=384,out_channels=64,kernel_size=1)
#         self.conv8_2 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1)
#         self.conv8_3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3, padding=1)
#         self.bn8_1 = nn.BatchNorm2d(64)
#         self.bn8_2 = nn.BatchNorm2d(256)
#         self.bn8_3 = nn.BatchNorm2d(256)

#         # 9
#         self.conv9_1 = nn.Conv2d(in_channels=512,out_channels=64,kernel_size=1)
#         self.conv9_2 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1)
#         self.conv9_3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3, padding=1)
#         self.bn9_1 = nn.BatchNorm2d(64)
#         self.bn9_2 = nn.BatchNorm2d(256)
#         self.bn9_3 = nn.BatchNorm2d(256)

#         #self.conv10 = nn.Conv2d(in_channels=512,out_channels=10,kernel_size=1)

#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
#         self.fc = nn.Linear(in_features=512, out_features=100) # adding this to simplify the output shape

#     def forward(self,x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.max_pool2d(x,2)

#         x = F.relu(self.bn2_1(self.conv2_1(x)))
#         x1 = F.relu(self.bn2_2(self.conv2_2(x)))
#         x2 = F.relu(self.bn2_3(self.conv2_3(x)))
#         x = torch.cat((x1,x2),dim = 1)

#         x = F.relu(self.bn3_1(self.conv3_1(x)))
#         x1 = F.relu(self.bn3_2(self.conv3_2(x)))
#         x2 = F.relu(self.bn3_3(self.conv3_3(x)))
#         x = torch.cat((x1,x2),dim = 1)

#         x = F.relu(self.bn4_1(self.conv4_1(x)))
#         x1 = F.relu(self.bn4_2(self.conv4_2(x)))
#         x2 = F.relu(self.bn4_3(self.conv4_3(x)))
#         x = torch.cat((x1,x2),dim = 1)
#         x = F.max_pool2d(x,2)

#         x = F.relu(self.bn5_1(self.conv5_1(x)))
#         x1 = F.relu(self.bn5_2(self.conv5_2(x)))
#         x2 = F.relu(self.bn5_3(self.conv5_3(x)))
#         x = torch.cat((x1,x2),dim = 1)

#         x = F.relu(self.bn6_1(self.conv6_1(x)))
#         x1 = F.relu(self.bn6_2(self.conv6_2(x)))
#         x2 = F.relu(self.bn6_3(self.conv6_3(x)))
#         x = torch.cat((x1,x2),dim = 1)

#         x = F.relu(self.bn7_1(self.conv7_1(x)))
#         x1 = F.relu(self.bn7_2(self.conv7_2(x)))
#         x2 = F.relu(self.bn7_3(self.conv7_3(x)))
#         x = torch.cat((x1,x2),dim = 1)

#         x = F.relu(self.bn8_1(self.conv8_1(x)))
#         x1 = F.relu(self.bn8_2(self.conv8_2(x)))
#         x2 = F.relu(self.bn8_3(self.conv8_3(x)))
#         x = torch.cat((x1,x2),dim = 1)
#         x = F.max_pool2d(x,2)

#         x = F.relu(self.bn9_1(self.conv9_1(x)))
#         x1 = F.relu(self.bn9_2(self.conv9_2(x)))
#         x2 = F.relu(self.bn9_3(self.conv9_3(x)))
#         x = torch.cat((x1,x2),dim = 1)

#         #x = F.relu((self.conv10(x)))
#         x = self.avgpool(x).view(-1, 512)
#         x = self.fc(x)

#         return F.log_softmax(x, dim=1)


class squeezenet(nn.Module):
    def __init__(self):
        super(squeezenet,self).__init__()

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
        self.fc = nn.Linear(in_features=512, out_features=100) # adding this to simplify the output shape

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
