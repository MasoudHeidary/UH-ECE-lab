# dataset	network 	params	top1 err	    top5 err	epoch(lr = 0.1)	epoch(lr = 0.02)	epoch(lr = 0.004)	epoch(lr = 0.0008)	total epoch
# cifar100	resnet18	11.2M	24.39	        6.95	    60	            60	                40	                40	                200


import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy

from globalVariable import *
from tool_d.log import Log

l = Log(LOG_FILE_NAME, terminal=True)

##################################################### data set
torch.set_printoptions(linewidth=100)
torch.set_grad_enabled(True)
torch.manual_seed(1)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

train_set = None
test_set = None


if DATA_SET == "Cifar100":
    train_set = torchvision.datasets.CIFAR100(
        root='./data/cifar100', #add your path here
        train=True,
        download=True,
        transform=transforms.Compose([transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(), normalize, transforms.RandomErasing(),])
    )
    test_set = torchvision.datasets.CIFAR100(
        root='./data/cifar100', #add your path here
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize])
    )

else:
    raise ValueError("DATASET INVALID")


def get_num_correct(pred, labels):
    return pred.argmax(dim=1).eq(labels).sum().item()

class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_list, last_epoch=-1):
        self.LR = lr_list
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [lr for lr in self.LR]


def train(network, save, lr, weight_decay, epoch_range):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, pin_memory=True)
    optimizer = optim.SGD(network.parameters(), lr=lr[0], momentum=spc_momentum,  weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    scheduler = CustomLRScheduler(optimizer, spc_lr)

    acc_train = []
    acc_test = []

    for epoch in range(epoch_range):
        total_loss = 0
        total_correct = 0
        network.train()

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

        l.println(f"training [{epoch:03}]: {total_correct/len(train_set) :.4f} \tLR:{optimizer.param_groups[0]['lr']}")
        acc_train.append(float(total_correct) / len(train_set))

        ### Testing ###
        correct_test = 0
        for batch_test in test_loader:  # Get batch
            images_test, labels_test = batch_test
            images_test, labels_test = images_test.to(device), labels_test.to(device)

            preds_test = network(images_test)  # pass batch to network
            correct_test += get_num_correct(preds_test, labels_test)
        l.println(f"testing accuracy: {correct_test / len(test_set)}")
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
    l.println(f"testing accuracy: {correct_test / len(test_set) :.4f}")
#####################################################


##################################################### MAIN
from tool import manipualte_percentage
import resnetCifar100





training_lst = []
inference_lst = []
if DATA_SET == 'Cifar100':
    training_lst = [
        {'network': resnetCifar100.resnet18(), 'path': './modeloutput/spcMod/resnet18Cifar100.pt', },
        # {'network': darknetCifar100(), 'path': './modeloutput/darknetCifar100.pt', },
        # {'network': squeezenetCifar100(), 'path': './modeloutput/squeezenetCifar100.pt', },
        # {'network': resnet18Cifar100(), 'path': './modeloutput/resnet18Cifar100.pt', },
        # {'network': vgg11Cifar100(), 'path': './modeloutput/vgg11Cifar100.pt', },
        # {'network': vgg19Cifar100(), 'path': './modeloutput/vgg19Cifar100.pt', },
    ]
    inference_lst = [
        # {'network': darknetManiCifar100(), 'path': './modeloutput/darknetCifar100.pt', },
        # {'network': squeezenetManiCifar100(), 'path': './modeloutput/squeezenetCifar100.pt', },
        # {'network': resnet18ManiCifar100(), 'path': './modeloutput/resnet18Cifar100.pt', },
        # {'network': vgg11ManiCifar100(), 'path': './modeloutput/vgg11Cifar100.pt', },
        # {'network': vgg19ManiCifar100(), 'path': './modeloutput/vgg19Cifar100.pt', },
    ]
else:
    raise ValueError("DATASET INVALID")


def main_train():
    l.println("---MAIN TRAINING---")
    for i in training_lst:
        l.println(f"network: {i['network'].__class__.__name__}")

        network = i['network']
        network = network.to(device)

        train(network, 
              save=i.get('path') or False, 
              lr=i.get('lr') or spc_lr, 
              weight_decay=i.get('weight_decay') or spc_weight_decay,
              epoch_range=i.get('epoch_range') or spc_epoch_range
              )
        l.println()


def main_inference():
    l.println("---MAIN INFERENCE---")
    for i in inference_lst:
        l.println(f"network: {i['network'].__class__.__name__}")
        
        network = i['network']
        network = network.to(device)

        network.load_state_dict(torch.load(i['path']))

        for i in default_manipulate_range:
            manipualte_percentage.set(i/default_manipulate_divider/100)
            l.println(f"set manipulate percentage: {i/default_manipulate_divider}/100%")
            infer(network)
            # l.println()
        
        l.println()


if __name__ == "__main__":
    l.println()
    l.println("PROGRAM START")
    l.println(f"TRAIN FLAG: {TRAIN_FLAG}")
    l.println(f"DATA SET: {DATA_SET}")

    if TRAIN_FLAG:
        if CONFIRM_TO_TRAIN:
            if(input("Confirm to train (y/n): ").lower() in ['y', 'yes']):
                pass
            else:
                exit("train cancelled")
        main_train()
    else:
        main_inference()
    
    l.println("PROGRAM FINISHED")