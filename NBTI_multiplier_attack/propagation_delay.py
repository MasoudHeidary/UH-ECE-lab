

from msimulator.Multiplier import MPn_rew
from msimulator.bin_func import signed_b
from tool import NBTI_formula as BTI

from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body


def get_FA_delay(fa_alpha, temp, sec):
    tg1_alpha = max(fa_alpha[0], fa_alpha[1])
    tg1_vth = abs(BTI.Vth) + BTI.delta_vth(BTI.Vdef, temp, tg1_alpha, BTI.Tclk, sec)
    tg1_pb = pmos_vth_to_body(tg1_vth)

    tg2_alpha = max(fa_alpha[2], fa_alpha[3])
    tg2_vth = abs(BTI.Vth) + BTI.delta_vth(BTI.Vdef, temp, tg2_alpha, BTI.Tclk, sec)
    tg2_pb = pmos_vth_to_body(tg2_vth)

    return tgate_pb_to_delay(tg1_pb) + tgate_pb_to_delay(tg2_pb)

def array_multiplier_pd(A, B, bitlen, alpha, temp, sec):
    mp = MPn_rew(signed_b(A, bitlen), signed_b(B, bitlen), bitlen, rew_lst=[])
    mp.output

    fa_delay = [
        [get_FA_delay(alpha[fa_i][fa_j], temp, sec) for fa_j in range(bitlen)]
        for fa_i in range(bitlen-1)
    ]
    
    ps_matrix = [
            [0 for _ in range(bitlen)]
        for _ in range(bitlen-1)
    ]

    for fa_i in range(bitlen-1):
        for fa_j in range(bitlen):

            if mp.gfa[fa_i][fa_j].sum or mp.gfa[fa_i][fa_j].carry:
                previous_block = 0

                i, j = fa_i-1, fa_j+1
                if i >= 0 and j < bitlen:
                    d = ps_matrix[i][j]
                    previous_block = max(previous_block, d)

                i, j = fa_i, fa_j-1
                if fa_j >= 0:
                    d = ps_matrix[i][j]
                    previous_block = max(previous_block, d)

                ps_matrix[fa_i][fa_j] = fa_delay[fa_i][fa_j] + previous_block
                # print(f"[{fa_i}][{fa_j}]: {previous_block} -> {ps_matrix[fa_i][fa_j]}")
    
    # return ps_matrix
    return max(ps_matrix[-1])


def array_multiplier_error_rate(bitlen, alpha, temp, sec, max_ps_delay):
    limit = 2 ** (bitlen - 1)
    length = 2*limit

    error_counter = 0
    for A in range(-limit, limit):
        for B in range(-limit, limit):
            pd = array_multiplier_pd(A, B, bitlen, alpha, temp, sec)
            if pd > max_ps_delay:
                error_counter += 1
            
    return error_counter / length



    

# from alpha import get_alpha
# BIT_LEN = 8
# TEMP = 273.15 + 80
# t_sec = 50 *7 *24*60*60

# if True and (__name__ == "__main__"):
#     alpha = get_alpha(MPn_rew, BIT_LEN)
    
#     mm = 0
#     for A in range(100):
#         for B in range(100):
#             pd = array_multiplier_pd(A, B, BIT_LEN, alpha, TEMP, t_sec)
#             mm = max(mm, max(pd[-1]))
#             print(f"{A} {B} -> {mm}")

#     # pd = array_multiplier_pd(0b01111111, 0b01111111, BIT_LEN, alpha, TEMP, t_sec)
#     # mm = max(mm, max(pd[-1]))
#     # print(f"MAX: {mm}")

#     print(pd)
#     print(f"max: {max(pd[-1])}")

# if True and (__name__ == "__main__"):
#     from alpha import get_alpha
