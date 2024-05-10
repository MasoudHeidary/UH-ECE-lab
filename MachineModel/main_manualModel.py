import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from copy import deepcopy



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def get_num_correct(pred, labels):
    return pred.argmax(dim=1).eq(labels).sum().item()


def train(network, save=False, save_path = ""):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, pin_memory=True)
    optimizer = optim.SGD(network.parameters(), lr=0.1,  weight_decay=0.0005)
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

        print("epoch: ", epoch, "total_correct: ", total_correct, " total loss: ", total_loss)
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
        torch.save(network.state_dict(), save_path) # Add you path where you want to save
    
    print('best accuracy: ', max(acc_test))


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




########################################################################## MAIN
from manualModel import squeezenet
TRAIN_FLAG = False
model_path = './modeloutput/squeezenet.pt'

def main():
    network = squeezenet() # change this based on which model you want
    network = network.to(device)

    if TRAIN_FLAG:
        train(network, save=True, save_path=model_path) # use this line if you want to train
    else:
        network.load_state_dict(torch.load(model_path)) # load the model you want
        infer(network)


    
if __name__ == "__main__":
    main()
########################################################################## END MAIN
