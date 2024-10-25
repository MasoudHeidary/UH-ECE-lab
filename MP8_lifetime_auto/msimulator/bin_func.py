

def signed_b(num: int, bit_len: int):
    num_cpy = num
    if num < 0:
        num_cpy = 2**bit_len + num
    bit_num = list(map(int, reversed(format(num_cpy, f'0{bit_len}b'))))

    if (num > 0) and (bit_num[-1] != 0):
        raise OverflowError(f"number {num} can't fit in signed #{bit_len} bits")
    if (num < 0) and (bit_num[-1] != 1):
        raise OverflowError(f"number {num} can't fit in signed #{bit_len} bits")
    return bit_num

def reverse_signed_b(binary_list):
    binary_str = ''.join(map(str, reversed(binary_list)))
    num = int(binary_str, 2)

    if binary_list[-1] == 1:
        num = num - (2**len(binary_list))
    return num