

from tool.log import Log
from tool import NBTI_formula as BTI
from msimulator.bin_func import signed_b,reverse_signed_b
from msimulator.Multiplier import MPn_rew
from alpha import AlphaMultiprocess

from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body

from propagation_delay import array_multiplier_error_rate

import matplotlib.pyplot as plt
from datetime import datetime
import random


BIT_LEN = 8
TEMP = 273.15 + 80
ALPHA_VERIFICATION = False
log = Log(f"{__file__}.{BIT_LEN}.log", terminal=True)

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

"""magic attack only applies one pair of input repetively for maximized aging"""
def get_single_alpha(raw_mp, bit_len, A, B, log=False):
    
    # alpah structure as counter
    alpha_row = bit_len - 1
    alpha_index = bit_len
    alpha = [
        [
            [0 for _ in range(6)]
            for _ in range(alpha_index)
        ]
        for _ in range(alpha_row)
    ]

    # counting alpha
    a_bin = signed_b(A, bit_len)
    b_bin = signed_b(B, bit_len)
    mp = raw_mp(a_bin, b_bin, bit_len, [])
    mp.output

    for row in range(alpha_row):
        for index in range(alpha_index):
            for t in range(6):
                alpha[row][index][t] += (not mp.gfa[row][index].p[t])
    
    # alpha counter -> alpha probability
    random.seed(7)
    for row in range(alpha_row):
        for index in range(alpha_index):
            for t in range(6):
                if alpha[row][index][t] == 1:
                    alpha[row][index][t] = random.randrange(80, 90+1)/100
    return alpha
    
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

def get_worst_input_pair(raw_mp, bit_len, critical_path, temp, log=False):
    limit = 2 ** (bit_len-1)
    age_sec = 50 * 7 *24*60*60
    
    worst_pair = (0, 0)
    worst_delay = 0
    w_alpha = []
    for A in range(-limit, limit):
        for B in range(-limit, limit):
            alpha = get_single_alpha(raw_mp, bit_len, A, B, log=False)
            pd = get_MP_delay(critical_path, alpha, temp, age_sec)
            if log and (alpha != w_alpha):
                log.println(f"pair: {(A, B)}, delay: {pd}")
            
            if pd > worst_delay:
                worst_delay = pd
                worst_pair = (A, B)
                w_alpha = alpha  
    return worst_pair

def sort_rewiring(wiring):
    return sorted(wiring, key=lambda x: x[-1], reverse=True)


########################################################################################
################## MAIN
########################################################################################

"""
MAGIC attack worst input pair

output:
because we are in ideal case, if we only apply one input the
unbalance nature is same and we will get the same delay after time t'
"""
if False and (__name__ == "__main__"):
    log.println(f"RUNNING: MAGIC attack worst input for bit_len [{BIT_LEN}]")
    worst_pair = get_worst_input_pair(MPn_rew, BIT_LEN, CRITICAL_FA_lst, TEMP, log=log)
    log.println(f":\n{worst_pair}")

"""
error rate of wire combination
"""
if True and __name__ == "__main__":
    log.println(f"RUNNING: error rate bitlen [{BIT_LEN}], REW_LST [MAGIC]")

    alpha_notamper = get_alpha(MPn_rew, BIT_LEN, log=False, rew_lst=[], verify=False)
    # alpha = get_alpha(MPn_rew, BIT_LEN, log=False, rew_lst=REW_LST, verify=False)
    alpha_magic = get_single_alpha(MPn_rew, BIT_LEN, 0, 0, log=False)
    
    margin_t_sec = 199 *7 *24*60*60
    max_ps_delay = get_MP_delay(CRITICAL_FA_lst, alpha_notamper, TEMP, margin_t_sec)   # fixed margin
    
    res = []
    for t_week in range(200):
        t_sec = t_week *7 *24*60*60

        err_rate, max_seen_delay = array_multiplier_error_rate(BIT_LEN, alpha_magic, TEMP, t_sec, max_ps_delay)
        res.append(err_rate)
        
        log.println(f"REW [MAGIC] week [{t_week:03}], error rate: {err_rate:.4f}, max seen delay: {max_seen_delay:.3f}, max_allowed_delay: {max_ps_delay:.3f}")
    log.println(f"REW [MAGIC], raw error rate: \n{res}")

