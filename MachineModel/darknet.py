import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class darknetNormal(nn.Module):
    def __init__(self):
        super(darknetNormal,self).__init__()

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
    

class darknetNoBn(nn.Module):
    def __init__(self):
        super(darknetNoBn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3, padding=1)
        # torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        
        ### 2
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, padding=1)
        # torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        
        ### 3
        self.conv3_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1, padding=0)
        self.conv3_3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        # torch.nn.init.xavier_normal_(self.conv3_1.weight)
        # torch.nn.init.xavier_normal_(self.conv3_2.weight)
        # torch.nn.init.xavier_normal_(self.conv3_3.weight)
        torch.nn.init.kaiming_uniform_(self.conv3_1.weight)
        torch.nn.init.kaiming_uniform_(self.conv3_2.weight)
        torch.nn.init.kaiming_uniform_(self.conv3_3.weight)

        ### 4
        self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1, padding=0)
        self.conv4_3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        # torch.nn.init.xavier_normal_(self.conv4_1.weight)
        # torch.nn.init.xavier_normal_(self.conv4_2.weight)
        # torch.nn.init.xavier_normal_(self.conv4_3.weight)
        torch.nn.init.kaiming_uniform_(self.conv4_1.weight)
        torch.nn.init.kaiming_uniform_(self.conv4_2.weight)
        torch.nn.init.kaiming_uniform_(self.conv4_3.weight)

        ### 5
        self.conv5_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1, padding=0)
        self.conv5_3 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1, padding=1)
        self.conv5_5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        # torch.nn.init.xavier_normal_(self.conv5_1.weight)
        # torch.nn.init.xavier_normal_(self.conv5_2.weight)
        # torch.nn.init.xavier_normal_(self.conv5_3.weight)
        # torch.nn.init.xavier_normal_(self.conv5_4.weight)
        # torch.nn.init.xavier_normal_(self.conv5_5.weight)
        torch.nn.init.kaiming_uniform_(self.conv5_1.weight)
        torch.nn.init.kaiming_uniform_(self.conv5_2.weight)
        torch.nn.init.kaiming_uniform_(self.conv5_3.weight)
        torch.nn.init.kaiming_uniform_(self.conv5_4.weight)
        torch.nn.init.kaiming_uniform_(self.conv5_5.weight)

        ### 6
        self.conv6_1 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1, padding=0)
        self.conv6_3 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, padding=1)
        self.conv6_4 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1, padding=1)
        self.conv6_5 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, padding=1)
        # torch.nn.init.xavier_normal_(self.conv6_1.weight)
        # torch.nn.init.xavier_normal_(self.conv6_2.weight)
        # torch.nn.init.xavier_normal_(self.conv6_3.weight)
        # torch.nn.init.xavier_normal_(self.conv6_4.weight)
        # torch.nn.init.xavier_normal_(self.conv6_5.weight)
        torch.nn.init.kaiming_uniform_(self.conv6_1.weight)
        torch.nn.init.kaiming_uniform_(self.conv6_2.weight)
        torch.nn.init.kaiming_uniform_(self.conv6_3.weight)
        torch.nn.init.kaiming_uniform_(self.conv6_4.weight)
        torch.nn.init.kaiming_uniform_(self.conv6_5.weight)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=1024, out_features=10)
        # torch.nn.init.xavier_normal_(self.fc.weight)
        torch.nn.init.kaiming_uniform_(self.fc.weight)

    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.relu(self.conv5_4(x))
        x = F.relu(self.conv5_5(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        x = F.relu(self.conv6_3(x))
        x = F.relu(self.conv6_4(x))
        x = F.relu(self.conv6_5(x))

        x = self.avgpool(x).view(-1, 1024)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
 


# TODO: ...
class darknetNoBnManipulated(nn.Module):
    def __init__(self):
        super(darknetNoBnManipulated,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # torch.nn.init.xavier_normal_(self.conv1.weight)
        
        ### 2
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # torch.nn.init.xavier_normal_(self.conv2.weight)
        
        ### 3
        self.conv3_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1, padding=0)
        self.conv3_3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.bn3_3 = nn.BatchNorm2d(128)
        # torch.nn.init.xavier_normal_(self.conv3_1.weight)
        # torch.nn.init.xavier_normal_(self.conv3_2.weight)
        # torch.nn.init.xavier_normal_(self.conv3_3.weight)

        ### 4
        self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1, padding=0)
        self.conv4_3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.bn4_3 = nn.BatchNorm2d(256)
        # torch.nn.init.xavier_normal_(self.conv4_1.weight)
        # torch.nn.init.xavier_normal_(self.conv4_2.weight)
        # torch.nn.init.xavier_normal_(self.conv4_3.weight)
        
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
        # torch.nn.init.xavier_normal_(self.conv5_1.weight)
        # torch.nn.init.xavier_normal_(self.conv5_2.weight)
        # torch.nn.init.xavier_normal_(self.conv5_3.weight)
        # torch.nn.init.xavier_normal_(self.conv5_4.weight)
        # torch.nn.init.xavier_normal_(self.conv5_5.weight)

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
        # torch.nn.init.xavier_normal_(self.conv6_1.weight)
        # torch.nn.init.xavier_normal_(self.conv6_2.weight)
        # torch.nn.init.xavier_normal_(self.conv6_3.weight)
        # torch.nn.init.xavier_normal_(self.conv6_4.weight)
        # torch.nn.init.xavier_normal_(self.conv6_5.weight)

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
 


