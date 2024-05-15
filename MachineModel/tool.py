import torch
from globalVariable import *


class ManipulatePercentage:
    def __init__(self) -> None:
        self.percentage = 0.00
    def set(self, per):
        self.percentage = per

manipualte_percentage = ManipulatePercentage()

def manipulate(f):
    random_changes = torch.rand_like(f) < manipualte_percentage.percentage
    random_indices = torch.randint(0, len(delta), f.size()).to(device)
    random_deltas = delta[random_indices]
    f[random_changes] += random_deltas[random_changes]
    return f
