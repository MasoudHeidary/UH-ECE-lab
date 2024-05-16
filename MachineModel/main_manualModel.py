import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy

from globalVariable import *
from log import Log

l = Log("log.txt", terminal=True)

##################################################### data set
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
#####################################################


##################################################### training
def get_num_correct(pred, labels):
    return pred.argmax(dim=1).eq(labels).sum().item()


def train(network, save, lr=default_lr, weight_decay=default_weight_decay, epoch_range=default_epoch_range):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, pin_memory=True)
    optimizer = optim.SGD(network.parameters(), lr=lr,  weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    acc_train = []
    acc_test = []

    for epoch in range(epoch_range):
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

        l.println(f"training [{epoch:03}]: {total_correct/len(train_set)}")
        acc_train.append(float(total_correct) / len(train_set))

        ### Testing ###
        correct_test = 0
        for batch_test in test_loader:  # Get batch
            images_test, labels_test = batch_test
            images_test, labels_test = images_test.to(device), labels_test.to(device)

            preds_test = network(images_test)  # pass batch to network
            correct_test += get_num_correct(preds_test, labels_test)
        # l.println(f"testing accuracy: {correct_test / len(test_set)}")
        acc_test.append(deepcopy(float(correct_test) / len(test_set)))
        scheduler.step()
    
    if save:
        torch.save(network.state_dict(), save) # Add you path where you want to save
        l.println(f"TRAINING RESULT SAVED: {save}")
    
    # print('best accuracy: ', max(acc_test))
#####################################################

##################################################### Infer
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
    l.println(f"testing accuracy: {correct_test / len(test_set)}")
#####################################################

##################################################### MAIN
from tool import manipualte_percentage
from darknet import *
from squeezenet import *
from resnet18 import *
from vgg11 import *
from vgg19 import *

training_lst = [
    {'network': darknetCipher10(), 'path': './modeloutput/darknetCipher10.pt', },
    {'network': squeezenetCipher10(), 'path': './modeloutput/squeezenetCipher10.pt', },
    {'network': resnet18Cipher10(), 'path': './modeloutput/resnet18Cipher10.pt', },
    {'network': vgg11Cipher10(), 'path': './modeloutput/vgg11Cipher10.pt', },
    {'network': vgg19Cipher10(), 'path': './modeloutput/vgg19Cipher10.pt', },
]

inference_lst =  [
    {'network': darknetManiCipher10(), 'path': './modeloutput/darknetCipher10.pt', },
    {'network': squeezenetManiCipher10(), 'path': './modeloutput/squeezenetCipher10.pt', },
    {'network': resnet18ManiCipher10(), 'path': './modeloutput/resnet18Cipher10.pt', },
    {'network': vgg11ManiCiphar10(), 'path': './modeloutput/vgg11Cipher10.pt', },
    {'network': vgg19ManiCipher10(), 'path': './modeloutput/vgg19Cipher10.pt', },
]

def main_train():
    l.println("---MAIN TRAINING---")
    for i in training_lst:
        l.println(f"network: {i['network'].__class__.__name__}")

        network = i['network']
        network = network.to(device)

        train(network, 
              save=i.get('path') or False, 
              lr=i.get('lr') or default_lr, 
              weight_decay=i.get('weight_decay') or default_weight_decay,
              epoch_range=i.get('epoch_range') or default_epoch_range
              )
        l.println()


def main_inference():
    l.println("---MAIN INFERENCE---")
    for i in inference_lst:
        l.println(f"network: {i['network'].__class__.__name__}")
        
        network = i['network']
        network = network.to(device)

        network.load_state_dict(torch.load(i['path']))

        for i in range(0, default_manipulate_range+1, default_manipulate_step):
            manipualte_percentage.set(i/default_manipulate_divider/100)
            l.println(f"set manipulate percentage: {i/default_manipulate_divider}/100%")
            infer(network)
            # l.println()
        
        l.println()


if __name__ == "__main__":
    l.println()
    l.println("PROGRAM START")
    l.println(f"TRAIN FLAG: {TRAIN_FLAG}")

    if TRAIN_FLAG:
        main_train()
    else:
        main_inference()
    
    l.println("PROGRAM FINISHED")
##################################################### END MAIN
