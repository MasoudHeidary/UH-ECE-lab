

from msimulator.Multiplier import MPn_rew, Wallace_rew
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

def get_FA_delay_matrix(bitlen, alpha, temp, sec):
    fa_delay = [
        [get_FA_delay(alpha[fa_i][fa_j], temp, sec) for fa_j in range(bitlen)]
        for fa_i in range(bitlen-1)
    ]
    return fa_delay

def array_multiplier_pd(A, B, bitlen, fa_delay):
    mp = MPn_rew(signed_b(A, bitlen), signed_b(B, bitlen), bitlen, rew_lst=[])
    mp.output
    
    ps_matrix = [
            [0 for _ in range(bitlen)]
        for _ in range(bitlen-1)
    ]

    for fa_i in range(bitlen-1):
        for fa_j in range(bitlen):

            if mp.gfa[fa_i][fa_j].sum or mp.gfa[fa_i][fa_j].carry:
                previous_block = 0

                # gate at top
                i, j = fa_i-1, fa_j+1
                if i >= 0 and j < bitlen and mp.gfa[i][j].sum:
                    d = ps_matrix[i][j]
                    previous_block = max(previous_block, d)

                # gate at right
                i, j = fa_i, fa_j-1
                if j >= 0 and mp.gfa[i][j].carry:
                    d = ps_matrix[i][j]
                    previous_block = max(previous_block, d)

                ps_matrix[fa_i][fa_j] = fa_delay[fa_i][fa_j] + previous_block
                # print(f"[{fa_i}][{fa_j}]: {previous_block} -> {ps_matrix[fa_i][fa_j]}")
    
    return max(ps_matrix[-1])


def array_multiplier_error_rate(bitlen, alpha, temp, sec, max_ps_delay):
    limit = 2 ** (bitlen - 1)
    length = (2 * limit) ** 2
    fa_delay_matrix = get_FA_delay_matrix(bitlen, alpha, temp, sec)

    max_seen_delay = 0
    error_counter = 0
    for A in range(-limit, limit):
        for B in range(-limit, limit):
            pd = array_multiplier_pd(A, B, bitlen, fa_delay_matrix)
            if pd > max_ps_delay:
                error_counter += 1
            max_seen_delay = max(max_seen_delay, pd)

    return error_counter / length, max_seen_delay


def wallace_multiplier_pd(A, B, bitlen, fa_delay):
    mp = Wallace_rew(signed_b(A, bitlen), signed_b(B, bitlen), bitlen, rew_lst=[])
    mp.output
    
    ps_matrix = [
            [0 for _ in range(bitlen)]
        for _ in range(bitlen-1)
    ]
    
    for fa_i in range(bitlen-1):
        for fa_j in range(bitlen):
            
            if mp.gfa[fa_i][fa_j].sum or mp.gfa[fa_i][fa_j].carry:
                previous_block = 0
                
                # gate at top
                i, j = fa_i-1, fa_j+1
                if i >= 0 and j < bitlen and mp.gfa[i][j].sum:
                    d = ps_matrix[i][j]
                    previous_block = max(d, previous_block)
                
                # gate at top right
                i, j = fa_i-1, fa_j
                if i >=0 and mp.gfa[i][j].carry:
                    d = ps_matrix[i][j]
                    previous_block = max(d, previous_block)

                # gate at right (for last row)
                i, j = fa_i, fa_j-1
                if (i == bitlen-2) and j >= 0 and mp.gfa[i][j].carry:
                    d = ps_matrix[i][j]
                    previous_block = max(d, previous_block)
                
                ps_matrix[fa_i][fa_j] = fa_delay[fa_i][fa_j] + previous_block
    return max(ps_matrix[-1])


def wallace_multiplier_error_rate(bitlen, alpha, temp, sec, max_ps_delay):
    limit = 2 ** (bitlen - 1)
    length = (2*limit) ** 2
    fa_delay_matrix = get_FA_delay_matrix(bitlen, alpha, temp, sec)
    
    max_seen_delay = 0
    error_counter = 0
    for A in range(-limit, limit):
        for B in range(-limit, limit):
            pd = wallace_multiplier_pd(A, B, bitlen, fa_delay_matrix)
            if pd > max_ps_delay:
                error_counter += 1
            max_seen_delay = max(max_seen_delay, pd)
    return error_counter/length, max_seen_delay