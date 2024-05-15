import torch

MANIPULATE_PERCENTAGE = 0.00

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

delta = torch.tensor(
    [1, 0.5, 0.25, 0.125, 0.0675,
     -1, -0.5, -0.25, -0.125, -0.0675]
).to(device)
delta *= 2

def set_manipualte_percenrage(per):
    global MANIPULATE_PERCENTAGE
    MANIPULATE_PERCENTAGE = per

def manipulate(f):
    random_changes = torch.rand_like(f) < MANIPULATE_PERCENTAGE
    random_indices = torch.randint(0, len(delta), f.size()).to(device)
    random_deltas = delta[random_indices]
    f[random_changes] += random_deltas[random_changes]
    return f

def get_per():
    return MANIPULATE_PERCENTAGE