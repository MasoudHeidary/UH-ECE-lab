import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from copy import deepcopy


################################################################################### data set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)
torch.set_printoptions(linewidth=100)
torch.set_grad_enabled(True)
torch.manual_seed(1)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

train_set = torchvision.datasets.CIFAR10(
    root='./data/cifar10', #add your path here
    train=True,
    download=True,
    transform=transforms.Compose([transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip(),
                                  transforms.ToTensor(), normalize, transforms.RandomErasing(),])
)

test_set = torchvision.datasets.CIFAR10(
    root='./data/cifar10', #add your path here
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), normalize])
)
###################################################################################


################################################################################### training
def get_num_correct(pred, labels):
    return pred.argmax(dim=1).eq(labels).sum().item()


def train(network, save, lr, weight_decay):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, pin_memory=True)
    optimizer = optim.SGD(network.parameters(), lr=lr,  weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    acc_train = []
    acc_test = []

    for epoch in range(50):
        total_loss = 0
        total_correct = 0
        network.train()
        count_in = 0

        for batch in train_loader:  # Get batch
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            preds = network(images)  # pass batch to network
            # print('a', images.shape)
            correct = get_num_correct(preds, labels)
            loss = criterion(preds, labels)  # Calculate loss

            optimizer.zero_grad()
            loss.backward()  # Calculate gradients

            optimizer.step()  # Update weights
            preds = preds.float()
            loss = loss.float()

            total_loss += loss.item()
            total_correct += correct

        print(f"#network: {network.__class__.__name__}", "epoch: ", epoch, "total_correct: ", total_correct, " total loss: ", total_loss)
        print("training accuracy: ", total_correct / len(train_set))
        acc_train.append(float(total_correct) / len(train_set))

        ### Testing ###
        correct_test = 0
        for batch_test in test_loader:  # Get batch
            images_test, labels_test = batch_test
            images_test, labels_test = images_test.to(device), labels_test.to(device)

            preds_test = network(images_test)  # pass batch to network
            correct_test += get_num_correct(preds_test, labels_test)
        print("testing accuracy: ", correct_test / len(test_set))
        acc_test.append(deepcopy(float(correct_test) / len(test_set)))
        scheduler.step()
    
    if save:
        torch.save(network.state_dict(), save) # Add you path where you want to save
        print(f"SAVED on: {save}")
    
    print('best accuracy: ', max(acc_test))
###################################################################################

################################################################################### Infer
def infer(network):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, pin_memory=True)
    correct_test = 0
    #import time
    #start = time.time()
    for batch_test in test_loader:  # Get batch
        images_test, labels_test = batch_test
        images_test, labels_test = images_test.to(device), labels_test.to(device)

        preds_test = network(images_test)  # pass batch to network
        correct_test += get_num_correct(preds_test, labels_test)
    #print("time: ", time.time() - start)
    print("testing accuracy: ", correct_test / len(test_set))
###################################################################################


########################################################################## MAIN
from manualModel import *
from darknet import *
from vgg11 import *
TRAIN_FLAG = False
# network_list = [
#     {'network': darknet(), 'path': './modeloutput/darknet.pt', 'train_flag': TRAIN_FLAG},
#     # {'network': resnet18(), 'path': './modeloutput/resnet18.pt', 'train_flag': TRAIN_FLAG},
#     # {'network': squeezenet(), 'path': './modeloutput/squeezenet.pt', 'train_flag': TRAIN_FLAG},
#     # {'network': vgg11(), 'path': './modeloutput/vgg11.pt', 'train_flag': TRAIN_FLAG},
#     # {'network': vgg19(), 'path': './modeloutput/vgg19.pt', 'train_flag': TRAIN_FLAG},
# ]

training_lst = [
    {'network': darknetNoBn(), 'path': './modeloutput/darknetNoBn.pt', 'lr': 0.05, 'weight_decay': 0.0001},
    # {'network': vgg11NoBn(), 'path': './modeloutput/vgg11NoBn.pt', 'lr': 0.3, 'weight_decay': 0.0002},
]

inference_lst =  [
    # {'network': vgg11NoBnManipulated(), 'path': './modeloutput/vgg11NoBn.pt'},
    {'network': vgg11Manipulated(), 'path': './modeloutput/vgg11.pt'},
]


# def main():
#     for i in network_list:

#         print(f"#################### \
#               network: {i.get('network').__class__.__name__}\t\
#               TRAIN: {str(i['train_flag']).upper()}")
        
#         network = i.get('network')
#         network = network.to(device)

#         if i['train_flag']:
#             train(network, save=True, save_path=i.get('path'))
#         else:
#             network.load_state_dict(torch.load(i.get('path')))
#             infer(network)


def main_train():
    for i in training_lst:

        print(f"####################\n# network: {i['network'].__class__.__name__}")

        network = i['network']
        network = network.to(device)

        train(network, 
              save=i.get('path') or False, 
              lr=i.get('lr'), 
              weight_decay=i.get('weight_decay')
              )

def main_inference():
    for i in inference_lst:
        print(f"####################\n # network: {i['network'].__class__.__name__}\t")
        
        network = i['network']
        network = network.to(device)

        network.load_state_dict(torch.load(i['path']))
        infer(network)
    
if __name__ == "__main__":
    if TRAIN_FLAG:
        main_train()
        main_inference()
    else:
        # for i in range(1, 5):
            # set_manipualte_percenrage(i/100)
            # print(f"==> {i}/100 %")
        main_inference()
########################################################################## END MAIN
