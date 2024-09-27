import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights


from models.mani import manipualte_percentage

from log import Log

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# PATH = "checkpoint/resnet18/Wednesday_03_July_2024_10h_42m_56s/resnet18-200-regular.pth"
# PATH = "checkpoint/vgg11/Wednesday_17_July_2024_15h_07m_56s/vgg11-200-regular.pth"
# PATH = "checkpoint/vgg19/Thursday_18_July_2024_10h_57m_31s/vgg19-200-regular.pth"
# PATH = "checkpoint/squeezenet/Thursday_18_July_2024_14h_14m_23s/squeezenet-200-regular.pth"
PATH = "checkpoint/darknet/Thursday_18_July_2024_16h_28m_01s/darknet-200-regular.pth"

log = Log(PATH.split('/')[-1])

#################################################################
#################################################################

#################################################################
#################################################################


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            # images = images.cuda()
            images = images.to(device)
            # labels = labels.cuda()
            labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    # if args.gpu:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    log.println('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)
    # net.load_state_dict(torch.load(args.weights))
    net.load_state_dict(torch.load(PATH))
    # print(net)
    net.eval()

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    # iter_per_epoch = len(cifar100_training_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)


    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        # input_tensor = input_tensor.cuda()
        input_tensor = input_tensor.to(device)
    writer.add_graph(net, input_tensor)


    log.println(f"{PATH}")
    for i in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        manipualte_percentage.set(i/100)
        log.println(f"set manipulate percentage: {i}/100%")
        eval_training()

    writer.close()
