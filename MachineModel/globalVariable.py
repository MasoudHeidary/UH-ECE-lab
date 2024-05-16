import torch

default_epoch_range = 100
default_lr = 0.1
default_weight_decay = 0.0005

TRAIN_FLAG = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(1)


default_manipulate_range = 3
default_manipulate_step = 1
default_manipulate_divider = 1
delta = torch.tensor(
    [1, 0.5, 0.25, 0.125, 0.0675,
     -1, -0.5, -0.25, -0.125, -0.0675]
).to(device)
delta *= 2
# delta /= 2