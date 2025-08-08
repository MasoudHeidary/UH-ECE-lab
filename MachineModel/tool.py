import torch
from globalVariable import *


class ManipulatePercentage:
    def __init__(self) -> None:
        self.percentage = 0.00
    def set(self, per):
        if not (0 <= per <= 1):
            raise RuntimeError("per should be in [0:1] range")
        self.percentage = per

manipualte_percentage = ManipulatePercentage()

# def manipulate(f):
#     random_changes = torch.rand_like(f) < manipualte_percentage.percentage
#     random_indices = torch.randint(0, len(delta), f.size()).to(device)
#     random_deltas = delta[random_indices]
#     f[random_changes] += random_deltas[random_changes]
#     return f

def map_bit_to_mantissa(bit_len):
    if bit_len == 16:
        raise RuntimeError("INVALID bitlen")
    elif bit_len == 10:
        mantissa = 5
    elif bit_len == 8:
        mantissa = 3
    elif bit_len == 6:
        mantissa = 1
    else:
        raise ValueError("Unsupported format")
    return mantissa

def quantize_mantissa(tensor, mantissa_bits):
    if mantissa_bits >= 7:
        return tensor
    steps = 2 ** mantissa_bits
    quantized = torch.round(tensor * steps) / steps
    return quantized

# def inject_msb_error(f, percentage):
#     if BIT_LENGTH >= 32:
#         return f
#     f_flat = f.flatten().clone()

#     k = int(len(f_flat) * percentage)
#     topk_indices = torch.topk(f_flat.abs(), k).indices
#     original_values = f_flat[topk_indices]
#     f_view = original_values.clone().detach().view(torch.float32)
#     int_view = f_view.view(torch.int32)

#     # IEEE754: bit 23 is the MSB of mantissa in float32
#     # MSB_mask = (1 << 22) | (1 << 21) | (1 << 21) | (1 << 20) | (1 << 19)
#     MSB_mask = 0
#     for i in range(23 - map_bit_to_mantissa(BIT_LENGTH)):
#         MSB_mask |= (1 << i)
        
#     int_view &= (~MSB_mask)
#     f_flat[topk_indices] = int_view.view(torch.float32)
#     return f_flat.view(f.shape)


def mantissa_msb_drop(f_tensor):
    f_tensor = f_tensor.clone()

    f_int = f_tensor.view(torch.int32)
    mantissa = f_int & 0x7FFFFF

    msb_pos = torch.floor(torch.log2(mantissa.float() + 1e-8)).int()  # avoid log2(0)
    shift_amounts = 23 - msb_pos  # shift left to drop MSB

    shifted = (mantissa << shift_amounts) & 0x7FFFFF
    new_f_int = (f_int & 0xFF800000) | shifted
    return new_f_int.view(torch.float32)

def mantissa_random(f_tensor):
    f_tensor = f_tensor.clone()
    device = f_tensor.device
    f_int = f_tensor.view(torch.int32)
    sign_and_exponent = f_int & 0xFF800000  # top 9 bits

    random_mantissa = torch.randint(
        low=0,
        high=2**23,
        size=f_tensor.shape,
        dtype=torch.int32,
        device=device
    )

    new_f_int = sign_and_exponent | random_mantissa
    return new_f_int.view(torch.float32)

def inject_msb_error(f, percentage):
    f_flat = f.flatten().clone()

    # multiplier accuracy drop
    if (BIT_LENGTH < 16):
        multiplier_mask = 0
        for i in range(23 - map_bit_to_mantissa(BIT_LENGTH)):
            multiplier_mask |= (1 << i)
        int_all = f_flat.view(torch.float32).view(torch.int32)
        int_all &= (~multiplier_mask)
        f_flat = int_all.view(torch.float32)

    # Apply extra MSB masking to top percentage
    if percentage > 0:
        k = int(len(f_flat) * percentage)
        topk_indices = torch.topk(f_flat.abs(), k).indices
        original_values = f_flat[topk_indices]

        f_view = original_values.clone().detach().view(torch.float32)
        int_view = f_view.view(torch.int32)

        # Randomly flip bits in the mantissa (bits 0 to 22)
        # for _ in range(map_bit_to_mantissa(BIT_LENGTH)):
        # for _ in range(1):
            # bit_positions = torch.randint(22-map_bit_to_mantissa(BIT_LENGTH), 23+1, size=int_view.shape, dtype=torch.int32).to(device)
            # bit_positions = torch.randint(0, 23+1, size=int_view.shape, dtype=torch.int32).to(device)
            # masks = 1 << bit_positions
            # int_view &= masks
            
        # random_masks = torch.randint(1 << (22-map_bit_to_mantissa(BIT_LENGTH)), 1 << (23+1), size=int_view.shape, dtype=torch.int32).to(device)
        random_masks = torch.randint(1 << (23-map_bit_to_mantissa(BIT_LENGTH)), (1<<24)-1, size=int_view.shape, dtype=torch.int32).to(device)
        int_view ^= random_masks
        # int_view &= 0xFFF80000
                                         

        corrupted_values = int_view.view(torch.float32)
        f_flat[topk_indices] = corrupted_values
    
    # if percentage > 0:
    #     k = int(len(f_flat) * percentage)
    #     topk_indices = torch.topk(f_flat, k).indices
    #     f_flat[topk_indices] = mantissa_msb_drop(f_flat[topk_indices])
        
    return f_flat.view(f.shape)





def manipulate(f):
    return inject_msb_error(f, manipualte_percentage.percentage)
