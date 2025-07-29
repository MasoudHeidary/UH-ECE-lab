

"""
"""

from tool.log import Log
from tool import NBTI_formula as BTI
from msimulator.bin_func import signed_b,reverse_signed_b
from msimulator.Multiplier import MPn_rew
from alpha_shrinked import AlphaSampled

from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body

from propagation_delay import array_multiplier_error_rate

import matplotlib.pyplot as plt
from datetime import datetime

BIT_LEN = 16
TEMP = 273.15 + 80
ALPHA_VERIFICATION = False
ALPHA_SAMPLE_COUNT = 50_000_000
RND_SEED = 7
PROCESS_COUNT = 20
log = Log(f"{__file__}.{BIT_LEN}.{TEMP}.log", terminal=True)

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
    return AlphaSampled(raw_mp, bit_len, rew_lst).run_multi(ALPHA_SAMPLE_COUNT, RND_SEED, log=log, proc_count=PROCESS_COUNT)

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
        log = False,
        bitlen = BIT_LEN,
        temp = TEMP,
        mp = MPn_rew,
        critical_fa_lst = CRITICAL_FA_lst,
):
    # default wiring in multiplier
    best_wiring = [fa + ('A', 'B', 'C', 0) for fa in critical_fa_lst]
    worst_wiring = [fa + ('A', 'B', 'C', 0) for fa in critical_fa_lst]

    default_alpha = get_alpha(mp, bitlen, log=log, rew_lst=[])
    if log:
        log.println(f"\n{default_alpha}")
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
    if log:
        log.println(f"alpha: \n{alpha}")
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


def sort_rewiring(wiring):
    return sorted(wiring, key=lambda x: x[-1], reverse=True)


########################################################################################
################## MAIN
########################################################################################


"""get critical path priorities"""
if False and (__name__ == "__main__"):
    """each path gets full rewiring -> aging(1 year) -> route with higher aging higher priority"""
    log.println("RUNNING: critical path priorities")
    for i in range(BIT_LEN - 1):
        critical_path = create_crit(i)
        _, worst_wiring = get_best_worst_wire_comb(
            log=False,
            bitlen=BIT_LEN,
            critical_fa_lst=critical_path,
            mp=MPn_rew,
        )

        alpha_nomitigation = get_alpha(MPn_rew, BIT_LEN, log=False, rew_lst=[])
        alpha_rewired = get_alpha(MPn_rew, BIT_LEN, log=False, rew_lst=worst_wiring)
        delay_nomitigation = get_MP_delay(critical_path, alpha_nomitigation, TEMP, 50 * 7 *24*60*60)
        delay_rewired = get_MP_delay(critical_path, alpha_rewired, TEMP, 50 * 7 *24*60*60)

        log.println(f"path [{i}] -> delay(1 year): {delay_nomitigation:.4f} -> {delay_rewired:.4f}")
        log.println(f"{sort_rewiring(worst_wiring)}")
    


"""specific wire combination aging"""
if False and (__name__ == "__main__"):
    # normal aging without mitigation

    # REW_LST = [(1, 4, 'A', 'C', 'B', 0.09658243544414136), (1, 5, 'A', 'B', 'C', 0), (2, 4, 'A', 'C', 'B', 0.1496285059359429), (2, 5, 'A', 'C', 'B', 0.06917656718668863), (3, 4, 'A', 'C', 'B', 0.18241572138802975), (3, 5, 'A', 'C', 'B', 0.12469209243091955), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (0, 1, 'A', 'C', 'B', 0.28976406320693054), (0, 2, 'A', 'C', 'B', 0.26259279119258666), (0, 3, 'A', 'C', 'B', 0.25481341220235604), (0, 4, 'A', 'C', 'B', 0.2315129781993061), (0, 5, 'B', 'C', 'A', 0.103645377172287), (4, 4, 'C', 'A', 'B', 0.10590474274168515), (4, 5, 'B', 'A', 'C', 0.07976750715539899)]
    REW_LST = []

    # circuit_size = (BIT_LEN - 1) * (BIT_LEN)

    #rewire top-20% of circuit
    # wire_comb = REW_LST[:(circuit_size//5)]
    wire_comb = REW_LST
    
    examine_wire_comb(
        # wire_comb=[(0, 1, 'A', 'C', 'B')], 
        # wire_comb=[], 
        wire_comb = wire_comb,
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
if True and (__name__ == "__main__"):
    best_wiring, worst_wiring = get_best_worst_wire_comb(log=log)
    log.println(f"worst wiring:\n{worst_wiring}")
    
    examine_multi_wire_comb(
        [worst_wiring, [], best_wiring],
        ["attack", "no-mitigation", "optimization"],
        log=log,
        plot="DELAY",
    )


"""
partial rewiring
"""
if False and (__name__ == "__main__"):
    _, worst_wiring = get_best_worst_wire_comb(log=False)
    worst_wiring = sorted(worst_wiring, key=lambda x: x[-1], reverse=True)

    full_combo = []
    full_combo_label = []
    # for combo in range(0, len(worst_wiring)+1, 4):
    for combo in [0, len(worst_wiring)//4, len(worst_wiring)//2, len(worst_wiring)*3//4, len(worst_wiring)]:
        full_combo.append(worst_wiring[0:combo])
        full_combo_label.append(f"{combo} / {len(worst_wiring)}")
    log.println(f"wire combo:\n{full_combo}")
    
    examine_multi_wire_comb(
        multi_wire_comb = full_combo,
        plot_labels = full_combo_label,
        log = log,
        plot = "DELAY"
    )



"""
rewiring list for all the FAs (sorted)
"""
if False and (__name__ == "__main__"):
    lst = []
    for fa_i in range(BIT_LEN - 1):
        for fa_j in range(BIT_LEN):
            lst += [(fa_i, fa_j)]

    best_wiring, worst_wiring = get_best_worst_wire_comb(critical_fa_lst=lst, log=False)
    log.println(f"worst complete wiring:\n{worst_wiring}")
    worst_wiring = sorted(worst_wiring, key=lambda x: x[-1], reverse=True)
    log.println(f"worst complete wiring (sorted):\n{worst_wiring}")


"""
error rate of wire combination
"""
if False and __name__ == "__main__":
    # REW_LST = []

    log.println(f"RUNNING: error rate bitlen [{BIT_LEN}], REW_LST [{len(REW_LST)}]: \n{REW_LST}")

    alpha_notamper = get_alpha(MPn_rew, BIT_LEN, log=False, rew_lst=[], verify=False)
    alpha = get_alpha(MPn_rew, BIT_LEN, log=False, rew_lst=REW_LST, verify=False)

    res = []
    for t_week in range(200):
    # for t_week in [50, 100, 150, 199]:
        t_sec = t_week *7 *25*60*60
        
        max_ps_delay = get_MP_delay(CRITICAL_FA_lst, alpha_notamper, TEMP, t_sec) * 1.10    #10% margin
        err_rate, max_seen_delay = array_multiplier_error_rate(BIT_LEN, alpha, TEMP, t_sec, max_ps_delay)
        res.append(err_rate)
        
        log.println(f"REW [{len(REW_LST)}] week [{t_week:03}], error rate: {err_rate:.3f}, max seen delay: {max_seen_delay:.3f}, max_allowed_delay: {max_ps_delay:.3f}")
    log.println(f"REW [{len(REW_LST)}], error rate: \n{res}")

