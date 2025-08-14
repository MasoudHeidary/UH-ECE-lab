
"""
paper:
Process Reliability Based Trojans through NBTI and HCI effects
"""

from tool.log import Log
from tool import NBTI_formula as BTI
from msimulator.bin_func import signed_b,reverse_signed_b
from msimulator.Multiplier import MPn_rew
from alpha import AlphaMultiprocess

from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body

# from propagation_delay import array_multiplier_error_rate
from propagation_delay import array_multiplier_pd

import matplotlib.pyplot as plt
from datetime import datetime
import random


vth_shift = 0.05

BIT_LEN = 8
TEMP = 273.15 + 80
ALPHA_VERIFICATION = False
log = Log(f"{__file__}.{BIT_LEN}.shift{vth_shift}.log", terminal=True)

########################################################################################
################## Critical Path
########################################################################################

"""range(bit_len-1)"""
def create_crit(i_th) -> list:
    crit = []
    for lay in range(1, BIT_LEN-2):
        crit += [(lay, i_th), (lay, i_th + 1)]
    crit += [(0, i) for i in range(0, i_th + 2)]
    crit += [(BIT_LEN-2, i) for i in range(i_th, BIT_LEN)]
    return crit

"""for multiplier propagation delay and optimization"""
"""all FA in first row + last 2 FA in each row """
CRITICAL_FA_lst = create_crit(BIT_LEN - 2)
log.println(f"Critical eFA list: {CRITICAL_FA_lst}")


########################################################################################
################## Functions
########################################################################################

def get_alpha(raw_mp, bit_len, log=False, rew_lst=[], verify=False):
    return AlphaMultiprocess(raw_mp, bit_len, log=log, rew_lst=rew_lst).run()

def seed_generator(sample_index):
    return 7*sample_index + 1

def generate_guassian_vth_base(bit_len, mu=0, sigma=0, base_vth=abs(BTI.Vth), seed=False):
    if seed:
        random.seed(seed)

    vth = [
        [[base_vth for _ in range(6)] for _ in range(bit_len)] for _ in range(bit_len-1)
    ]

    for fa_i in range(bit_len-1):
        for fa_j in range(bit_len):
            for t_index in range(6):
                vth_variation = random.gauss(mu, sigma)
                vth[fa_i][fa_j][t_index] *= (1 + vth_variation)
                
                vth[fa_i][fa_j][t_index] *= (1 + vth_shift)

    return vth

def get_FA_delay(fa_alpha, temp, sec):
    tg1_alpha = max(fa_alpha[0], fa_alpha[1])
    tg1_vth = abs(BTI.Vth) + BTI.delta_vth(BTI.Vdef, temp, tg1_alpha, BTI.Tclk, sec)
    tg1_pb = pmos_vth_to_body(tg1_vth)

    tg2_alpha = max(fa_alpha[2], fa_alpha[3])
    tg2_vth = abs(BTI.Vth) + BTI.delta_vth(BTI.Vdef, temp, tg2_alpha, BTI.Tclk, sec)
    tg2_pb = pmos_vth_to_body(tg2_vth)

    return tgate_pb_to_delay(tg1_pb) + tgate_pb_to_delay(tg2_pb)

def get_MP_delay(critical_fa_lst, alpha, temp, sec):
    ps = 0
    for fa_lay, fa_i in critical_fa_lst:
        ps += get_FA_delay(alpha[fa_lay][fa_i], temp, sec)
    return ps


def get_monte_FA_delay(fa_vth, fa_alpha, temp, sec):
    tg1_vth_1 = fa_vth[0] + BTI.delta_vth(BTI.Vdef, temp, fa_alpha[0], BTI.Tclk, sec)
    tg1_vth_2 = fa_vth[1] + BTI.delta_vth(BTI.Vdef, temp, fa_alpha[1], BTI.Tclk, sec)
    tg1_vth = max(tg1_vth_1, tg1_vth_2)
    tg1_pb = pmos_vth_to_body(tg1_vth)

    tg2_vth_1 = fa_vth[2] + BTI.delta_vth(BTI.Vdef, temp, fa_alpha[2], BTI.Tclk, sec)
    tg2_vth_2 = fa_vth[3] + BTI.delta_vth(BTI.Vdef, temp, fa_alpha[3], BTI.Tclk, sec)
    tg2_vth = max(tg2_vth_1, tg2_vth_2)
    tg2_pb = pmos_vth_to_body(tg2_vth)

    # //fa_vth[4] fa_vth[5] ignored
    return tgate_pb_to_delay(tg1_pb) + tgate_pb_to_delay(tg2_pb)

def get_FA_delay_matrix(bit_len, vth, alpha, temp, sec):
    fa_delay = [
        [get_monte_FA_delay(vth[fa_i][fa_j], alpha[fa_i][fa_j], temp, sec) for fa_j in range(bit_len)]
        for fa_i in range(bit_len - 1)
    ]
    return fa_delay

def get_monte_path_delay(vth_matrix, critical_fa_lst, alpha, temp, sec):
    ps = 0
    for fa_lay, fa_i in critical_fa_lst:
        ps += get_monte_FA_delay(vth_matrix[fa_lay][fa_i], alpha[fa_lay][fa_i], temp, sec)
    return ps

def get_monte_MP_delay(sample_id, path_lst, alpha, temp, sec):
    ps = 0
    vth_matrix = generate_guassian_vth_base(BIT_LEN, seed=seed_generator(sample_id))
    for path in path_lst:
        path_delay = get_monte_path_delay(vth_matrix, path, alpha, temp, sec)
        ps = max(ps, path_delay)
    return ps

def get_error_rate(bitlen, vth, alpha, temp, sec, max_ps_delay):
    limit = 2 ** (bitlen - 1)
    length = (2 * limit) ** 2

    error_counter = 0
    max_seen_delay = 0
    fa_delay = get_FA_delay_matrix(bitlen, vth, alpha, temp, sec)
    
    for A in range(-limit, limit):
        for B in range(-limit, limit):
            pd = array_multiplier_pd(A, B, bitlen, fa_delay)
            if pd > max_ps_delay:
                error_counter += 1
            max_seen_delay = max(max_seen_delay, pd)
    return error_counter / length, max_seen_delay

########################################################################################
################## MAIN
########################################################################################

"""
error rate of wire combination
"""
if True and __name__ == "__main__":
    log.println(f"RUNNING: error rate bitlen [{BIT_LEN}], REW_LST [PT]")

    alpha_notamper = get_alpha(MPn_rew, BIT_LEN, log=False, rew_lst=[], verify=False)
    
    margin_t_sec = 199 *7 *24*60*60
    max_ps_delay = get_MP_delay(CRITICAL_FA_lst, alpha_notamper, TEMP, margin_t_sec)   # fixed margin
    
    res = []
    vth = generate_guassian_vth_base(BIT_LEN, seed=False)
    for t_week in range(200):
        t_sec = t_week *7 *24*60*60

        err_rate, max_seen_delay = get_error_rate(BIT_LEN, vth, alpha_notamper, TEMP, t_sec, max_ps_delay)
        res.append(err_rate)
        
        log.println(f"REW [PT] week [{t_week:03}], error rate: {err_rate:.4f}, max seen delay: {max_seen_delay:.3f}, max_allowed_delay: {max_ps_delay:.3f}")
    log.println(f"REW [PT], raw error rate: \n{res}")

