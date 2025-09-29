
import torch
import torch.nn as nn
from itertools import product

from config import DEVICE


# ============================================================
# ======== list of presentable numbers in different standards
# ============================================================
def floating_possible_presentation(exp, frac):
    def fraction(bit):
        sum = 1
        position_value = 0.5
        for i in bit:
            sum += (i * position_value)
            position_value *= 0.5
        return sum

    def exponent(bit):
        position_value = 2 ** (len(bit) - 1)
        ex = 1 - position_value
        for i in bit:
            ex += (i * position_value)
            position_value //= 2
        return ex

    res = []
    for e in product([0,1], repeat=exp):
        for f in product([0,1], repeat=frac):
            q = 2**exponent(e) * fraction(f)
            # print(f"{e}, {f} = {q}")
            res += [q]
    return res

def fixed_point_possible_presentation(int, frac):
    def integer(bit):
        sum = 0
        position_value = 2 ** (len(bit) - 1)
        for i in bit:
            sum += (i * position_value)
            position_value //= 2
        return sum
    
    def fraction(bit):
        sum = 0
        position_value = 0.5
        for i in bit:
            sum += (i * position_value)
            position_value *= 0.5
        return sum
    
    res = []
    for i in product([0,1], repeat=int):
        for f in product([0,1], repeat=frac):
            q = integer(i) + fraction(f)
            # print(f"{e}, {f} = {q}")
            res += [q]
    return res


# ============================================================
# ======== Quantization map & functions
# ============================================================

# INT 1.0.3
INT4_LEVELS = torch.tensor([    
    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875
    ], device=torch.device(DEVICE), dtype=torch.float32)
INT4_LEVELS = torch.cat((INT4_LEVELS, -INT4_LEVELS))

# INT 1.0.7
INT8_LEVELS = torch.tensor([
    0.0, 0.0078125, 0.015625, 0.0234375, 0.03125, 0.0390625, 0.046875, 0.0546875, 0.0625, 
    0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.1328125, 
    0.140625, 0.1484375, 0.15625, 0.1640625, 0.171875, 0.1796875, 0.1875, 0.1953125, 0.203125, 
    0.2109375, 0.21875, 0.2265625, 0.234375, 0.2421875, 0.25, 0.2578125, 0.265625, 0.2734375, 
    0.28125, 0.2890625, 0.296875, 0.3046875, 0.3125, 0.3203125, 0.328125, 0.3359375, 0.34375, 
    0.3515625, 0.359375, 0.3671875, 0.375, 0.3828125, 0.390625, 0.3984375, 0.40625, 0.4140625, 
    0.421875, 0.4296875, 0.4375, 0.4453125, 0.453125, 0.4609375, 0.46875, 0.4765625, 0.484375, 
    0.4921875, 0.5, 0.5078125, 0.515625, 0.5234375, 0.53125, 0.5390625, 0.546875, 0.5546875, 
    0.5625, 0.5703125, 0.578125, 0.5859375, 0.59375, 0.6015625, 0.609375, 0.6171875, 0.625, 
    0.6328125, 0.640625, 0.6484375, 0.65625, 0.6640625, 0.671875, 0.6796875, 0.6875, 0.6953125, 
    0.703125, 0.7109375, 0.71875, 0.7265625, 0.734375, 0.7421875, 0.75, 0.7578125, 0.765625, 
    0.7734375, 0.78125, 0.7890625, 0.796875, 0.8046875, 0.8125, 0.8203125, 0.828125, 0.8359375, 
    0.84375, 0.8515625, 0.859375, 0.8671875, 0.875, 0.8828125, 0.890625, 0.8984375, 0.90625, 
    0.9140625, 0.921875, 0.9296875, 0.9375, 0.9453125, 0.953125, 0.9609375, 0.96875, 0.9765625, 
    0.984375, 0.9921875
    ], device=torch.device(DEVICE), dtype=torch.float32)
INT8_LEVELS = torch.cat((INT8_LEVELS, -INT8_LEVELS))

# FTP 1.3.0
FTP4_LEVELS = torch.tensor([   
    0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16
    ], device=torch.device(DEVICE), dtype=torch.float32
)
FTP4_LEVELS = torch.cat((FTP4_LEVELS, -FTP4_LEVELS))

# FTP 4.1
FTP6_LEVELS = torch.tensor([    
    0, 0.0078125, 0.01171875, 0.015625, 0.0234375, 0.03125, 0.046875, 0.0625, 0.09375, 
    0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 
    16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0
    ], device=torch.device(DEVICE), dtype=torch.float32
)
FTP6_LEVELS = torch.cat((FTP6_LEVELS, -FTP6_LEVELS))

# FTP 1.4.3
FTP8_LEVELS = torch.tensor([    
    0, 0.0078125, 0.0087890625, 0.009765625, 0.0107421875, 0.01171875, 0.0126953125, 0.013671875,
    0.0146484375, 0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 
    0.02734375, 0.029296875, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 
    0.05078125, 0.0546875, 0.05859375, 0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 
    0.1015625, 0.109375, 0.1171875, 0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 
    0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 
    0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5, 
    1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 
    6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 
    24.0, 26.0, 28.0, 30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 72.0, 
    80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 
    224.0, 240.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0
    ], device=torch.device(DEVICE), dtype=torch.float32
)
FTP8_LEVELS = torch.cat((FTP8_LEVELS, -FTP8_LEVELS))


def map_tensor(tensor: torch.tensor, levels: torch.tensor, min_noise=0, max_noise=0):
    flat = tensor.reshape(-1, 1)
    dist = (flat - levels[None, :]).abs()
    indices = dist.argmin(dim=-1)
    quantized = levels[indices].reshape(tensor.shape)
    
    if max_noise != 0:
        noise_factor = torch.empty_like(quantized).uniform_(1 + min_noise, 1 + max_noise)
        return (quantized * noise_factor)
    else:
        return quantized

def float16_tensor(tensor: torch.tensor, min_noise=0, max_noise=0):
    quantized = tensor.half().float()
    noise_factor = torch.empty_like(quantized).uniform_(1 + min_noise, 1 + max_noise)
    return (quantized * noise_factor)
    


def ftp_modify_tensor(tensor: torch.tensor, precision):
    if precision == "ftp32":
        return tensor
    
    if precision == "ftp16":
        return float16_tensor(tensor, 0, 0.01)
    
    if precision == "ftp8":
        return map_tensor(tensor, FTP8_LEVELS, 0, 0.03)
    
    if precision == "int8":
        return map_tensor(tensor, INT8_LEVELS, 0, 0)
    
    if precision == "ftp4":
        return map_tensor(tensor, FTP4_LEVELS, 0, 0)
    
    if precision == "int4":
        return map_tensor(tensor, INT4_LEVELS, 0, 0)
    
    raise ValueError("invalid precision")

def ftp_modify_model(model, precision):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight.copy_(ftp_modify_tensor(module.weight, precision))
                if module.bias is not None:
                    module.bias.copy_(ftp_modify_tensor(module.bias, precision))
        else:
            ftp_modify_model(module, precision)
    return model
