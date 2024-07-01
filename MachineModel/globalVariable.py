import torch

default_epoch_range = 200
default_lr = 0.5
default_weight_decay = 0.0005

'Cifar10'
'Cifar100'
'SVHN'
DATA_SET = 'Cifar100'

TRAIN_FLAG = True
CONFIRM_TO_TRAIN = True

LOG_FILE_NAME = False or "CIFAR100-LR-0.5.txt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)

default_manipulate_range = range(0, 5+1, 1)
default_manipulate_divider = 1
delta = torch.tensor(
    [1, 0.5, 0.25, 0.125, 0.0675,
     -1, -0.5, -0.25, -0.125, -0.0675]
).to(device)
delta *= 2
# delta /= 2
# delta /= 2