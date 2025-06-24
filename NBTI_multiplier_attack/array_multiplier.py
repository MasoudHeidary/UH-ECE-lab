

from tool.log import Log
from tool import NBTI_formula as BTI
from msimulator.bin_func import signed_b,reverse_signed_b
from msimulator.Multiplier import MPn_rew
from alpha import AlphaMultiprocess

from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body

import matplotlib.pyplot as plt
from datetime import datetime

ALPHA_VERIFICATION = False

BIT_LEN = 8
TEMP = 273.15 + 30
log = Log(f"{__file__}.{BIT_LEN}.{TEMP}.log", terminal=True)


########################################################################################
################## Critical Path
########################################################################################

"""for multiplier propagation delay and optimization"""
CRITICAL_FA_lst = []

"""all FA in first row + last 2 FA in each row """
CRITICAL_FA_lst += [(0, i) for i in range(BIT_LEN)]
for lay in range(1, BIT_LEN-1):
    CRITICAL_FA_lst += [(lay, BIT_LEN-2), (lay, BIT_LEN-1)]
log.println(f"Critical eFA list: {CRITICAL_FA_lst}")

"""first 2 FA in each row + all FA in last row"""
# for lay in range(0, BIT_LEN-2):
#     CRITICAL_FA_lst += [(lay, 0), (lay, 1)]
# CRITICAL_FA_lst += [(BIT_LEN-2, i) for i in range(BIT_LEN)]
# log.println(f"Critical eFA list: {CRITICAL_FA_lst}")


########################################################################################
################## Functions
########################################################################################

def get_alpha(raw_mp, bit_len, log=False, rew_lst=[], verify=False):
    return AlphaMultiprocess(raw_mp, bit_len, log=log, rew_lst=rew_lst).run()

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


def get_best_worst_wire_comb(
        log = log,
        bitlen = BIT_LEN,
        temp = TEMP,
        mp = MPn_rew,
        critical_fa_lst = CRITICAL_FA_lst,
):
    # default wiring in multiplier
    best_wiring = [fa + ('A', 'B', 'C', 0) for fa in critical_fa_lst]
    worst_wiring = [fa + ('A', 'B', 'C', 0) for fa in critical_fa_lst]

    default_alpha = get_alpha(mp, bitlen, log=False, rew_lst=[])
    # log.println(f"{default_alpha}")
    if log:
        log.println(f"default alpha done")


    # iterate in Critical FA list, and log the worst wiring for each
    for fa_index, fa in enumerate(critical_fa_lst):
        wire_combination = [
            ('A', 'B', 'C'),
            ('A', 'C', 'B'),
            ('B', 'A', 'C'),
            ('B', 'C', 'A'),
            ('C', 'A', 'B'),
            ('C', 'B', 'A'),
        ]
        lay, i = fa
        FA_zero_delay = get_FA_delay(default_alpha[lay][i], temp, 0)

        aging_period = 12*30 *24*60*60
        fa_default_delay = get_FA_delay(default_alpha[lay][i], temp, aging_period)
        fa_default_aging_rate = (fa_default_delay - FA_zero_delay) / FA_zero_delay
        if log:
            log.println(f"default wiring, delay rate: {fa_default_aging_rate * 100 :.2f}% [t:{aging_period}s]")
        
        _worst_rate = fa_default_aging_rate
        _best_rate = fa_default_aging_rate
        for comb in wire_combination[1:]:
            rewire = fa + comb
            rewire_alpha = get_alpha(mp, bitlen, log=False, rew_lst=[rewire])

            fa_delay = get_FA_delay(rewire_alpha[lay][i], temp, aging_period)
            fa_aging_rate = (fa_delay - FA_zero_delay) / FA_zero_delay
            if log:
                log.println(f"{rewire}, delay rate: {fa_aging_rate * 100 :.2f}% [t:{aging_period}s]")

            if fa_aging_rate > _worst_rate:
                _worst_rate = fa_aging_rate
                worst_wiring[fa_index] = fa + comb + (fa_aging_rate - fa_default_aging_rate, )
                if log:
                    log.println(f"-new worst")
            if fa_aging_rate < _best_rate:
                _best_rate = fa_aging_rate
                best_wiring[fa_index] = fa + comb + (fa_aging_rate - fa_default_aging_rate, )
                if log:
                    log.println(f"-new best")
            
        if log:
            log.println()
    
    if log:
        log.println(f"best wiring combination: \n{best_wiring}")
        log.println(f"worst wiring combination: \n{worst_wiring}")

    return (best_wiring, worst_wiring)


"""
wire combination notation:

(0, 0, 'C', 'B', 'A', 0.50)
- FA index
- FA wiring combination
- difference of aging_rate and default_aging_rate
"""
def examine_wire_comb(
        wire_comb,
        bit_len = BIT_LEN, 
        temp = TEMP, 
        log = log, 
        plot = "RATE" or "DELAY", 
        plot_save_clear = True,
        plot_label = "",
        alpha_verification = ALPHA_VERIFICATION, 
        critical_fa_lst = CRITICAL_FA_lst, 
        mp = MPn_rew,
    ): 
    if log:   
        log.println(f"aging log for following wire comb \n{wire_comb}")
    
    alpha = get_alpha(mp, bit_len, log=False, rew_lst=wire_comb, verify=alpha_verification)
    _mp_zero_delay = get_MP_delay(critical_fa_lst, alpha, temp, 0)
    
    res_week = []
    res_delay = []

    for week in range(0, 200):
        delay = get_MP_delay(critical_fa_lst, alpha, temp, week * 7 *24*60*60)
        aging_rate = (delay - _mp_zero_delay) / _mp_zero_delay
        if log:
            log.println(f"week {week:03}: {delay: 8.2f} [{aging_rate * 100 :4.2f}%]")
        
        if plot:
            res_week.append(week)

            if plot == "DELAY":
                res_delay.append(delay)
            elif plot == "RATE" or True:
                res_delay.append(aging_rate)

    
    if plot:
        plt.plot(res_week, res_delay, label=plot_label)
        plt.title(f"ArrayMultiplier-BIT-{bit_len}-TEMP-{temp}")

        if plot_save_clear:
            timestamp = datetime.now().strftime('%m,%d-%H:%M:%S.%f')
            fig_name = f"fig-{timestamp}.jpg"
            plt.legend()
            plt.savefig(fig_name)
            plt.clf()
            if log:
                log.println(f"plot saved in {fig_name}")



def examine_multi_wire_comb(
        multi_wire_comb,
        plot_labels,
        log = log, 
        plot = True,
    ): 

    for i_sub_wc, sub_wc in enumerate(multi_wire_comb):
        examine_wire_comb(
            sub_wc,
            log = log, 
            plot = plot, 
            plot_save_clear = True if i_sub_wc==len(multi_wire_comb)-1 else False,
            plot_label = plot_labels[i_sub_wc],
        )


########################################################################################
################## MAIN
########################################################################################

"""comparing different critical path configuration"""
if False:
    PLOT_TYPE = "DELAY"
    # BIT_LEN = 8     ### GLOBAL OVERRIDE 

    # Critical path 1
    critical_fa_lst = []
    critical_fa_lst += [(0, i) for i in range(BIT_LEN)]
    for lay in range(1, BIT_LEN-1):
        critical_fa_lst += [(lay, BIT_LEN-2), (lay, BIT_LEN-1)]
    log.println(f"critical path 1: \n{critical_fa_lst}")
    
    examine_wire_comb(
        [], 
        plot=PLOT_TYPE, 
        plot_save_clear=False, 
        plot_label=f"crit 1 no-mitigation", 
        critical_fa_lst=critical_fa_lst
    )

    _, worst_wire_comb = get_best_worst_wire_comb(log=False, critical_fa_lst=critical_fa_lst)
    examine_wire_comb(
        worst_wire_comb, 
        plot=PLOT_TYPE, 
        plot_save_clear=False, 
        plot_label="Crit 1 attack", 
        critical_fa_lst=critical_fa_lst
    )


    # Critical path 2
    critical_fa_lst = []
    for lay in range(0, BIT_LEN-2):
        critical_fa_lst += [(lay, 0), (lay, 1)]
    critical_fa_lst += [(BIT_LEN-2, i) for i in range(BIT_LEN)]
    log.println(f"critical path 2: \n{critical_fa_lst}")
    examine_wire_comb(
        [], 
        plot=PLOT_TYPE, 
        plot_save_clear=False, 
        plot_label=f"Crit 2 no-mitigation", 
        critical_fa_lst=critical_fa_lst
    )

    _, worst_wire_comb = get_best_worst_wire_comb(log=False, critical_fa_lst=critical_fa_lst)
    examine_wire_comb(
        worst_wire_comb, 
        plot=PLOT_TYPE, 
        plot_save_clear=False, 
        plot_label="Crit 2 attack", 
        critical_fa_lst=critical_fa_lst
    )

    # Critical path 3
    critical_fa_lst = []
    for lay in range(1, BIT_LEN-2):
        critical_fa_lst += [(lay, 3), (lay, 4)]
    critical_fa_lst += [(0, i) for i in range(0, BIT_LEN//2 + 1)]
    critical_fa_lst += [(BIT_LEN-2, i) for i in range(BIT_LEN//2 - 1, BIT_LEN)]
    log.println(f"critical path 3: \n{critical_fa_lst}")

    examine_wire_comb(
        [], 
        plot=PLOT_TYPE, 
        plot_save_clear=False, 
        plot_label=f"Crit 3 no-mitigation", 
        critical_fa_lst=critical_fa_lst
    )

    _, worst_wire_comb = get_best_worst_wire_comb(log=False, critical_fa_lst=critical_fa_lst)
    examine_wire_comb(
        worst_wire_comb, 
        plot=PLOT_TYPE, 
        plot_save_clear=True, 
        plot_label="Crit 3 attack", 
        critical_fa_lst=critical_fa_lst
    )



"""specific wire combination aging"""
if False:
    # normal aging without mitigation
    examine_wire_comb(
        # wire_comb=[(0, 1, 'A', 'C', 'B')], 
        wire_comb=[], 
        bit_len=BIT_LEN, 
        temp=TEMP, 
        log=log, 
        plot=True, 
        alpha_verification=ALPHA_VERIFICATION,
        critical_fa_lst=CRITICAL_FA_lst
        )
    

"""
extracting best and worst wiring combination for the provided multiplier (6, 8, 10 bits)
"""
if True:
    best_wiring, worst_wiring = get_best_worst_wire_comb(log=False)
    log.println(f"{worst_wiring}")
    
    examine_multi_wire_comb(
        [worst_wiring, [], best_wiring],
        ["attack", "no-mitigation", "optimization"],
        log=log,
        plot="DELAY",
    )


"""
partial rewiring
"""
if False:
    _, worst_wiring = get_best_worst_wire_comb(log=False)
    worst_wiring = sorted(worst_wiring, key=lambda x: x[-1], reverse=True)

    full_combo = []
    full_combo_label = []
    for combo in range(0, len(worst_wiring)+1, 4):
        full_combo.append(worst_wiring[0:combo])
        full_combo_label.append(f"{combo} / {len(worst_wiring)}")
    
    examine_multi_wire_comb(
        multi_wire_comb = full_combo,
        plot_labels = full_combo_label,
        log = False,
        plot = "RATE"
    )