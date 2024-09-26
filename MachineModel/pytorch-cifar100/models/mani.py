import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

delta = torch.tensor(
    [1, 0.5, 0.25, 0.125, 0.0675,
     -1, -0.5, -0.25, -0.125, -0.0675]
).to(device)
# delta *= 2

class ManipulatePercentage:
    def __init__(self) -> None:
        self.percentage = 0.00
    def set(self, per):
        if not (0 <= per <= 1):
            raise RuntimeError("per should be in [0:1] range")
        self.percentage = per

manipualte_percentage = ManipulatePercentage()

def manipulate(f):
    random_changes = torch.rand_like(f) < manipualte_percentage.percentage
    random_indices = torch.randint(0, len(delta), f.size()).to(device)
    random_deltas = delta[random_indices]
    f[random_changes] += random_deltas[random_changes]
    return f