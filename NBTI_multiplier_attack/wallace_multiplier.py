

"""
BIT_LEN = 8 critical path priority:
path [0] -> delay(30240000s): 467.8 -> 565.5
[(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844)]
path [1] -> delay(30240000s): 467.8 -> 564.0
[(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 1, 'A', 'B', 'C', 0)]
path [2] -> delay(30240000s): 467.6 -> 557.0
[(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (1, 1, 'A', 'C', 'B', 0.23806349924964598), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 2, 'A', 'B', 'C', 0)]
path [3] -> delay(30240000s): 467.5 -> 549.5
[(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (1, 2, 'A', 'C', 'B', 0.23806349924964598), (2, 1, 'A', 'C', 'B', 0.22623808445886484), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 3, 'A', 'B', 'C', 0)]
path [4] -> delay(30240000s): 467.4 -> 541.5
[(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (1, 3, 'A', 'C', 'B', 0.23806349924964598), (2, 2, 'A', 'C', 'B', 0.22623808445886484), (3, 1, 'A', 'C', 'B', 0.21645413617042697), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 4, 'A', 'B', 'C', 0)]
path [5] -> delay(30240000s): 467.4 -> 533.3
[(6, 0, 'C', 'A', 'B', 0.559487579332143), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (1, 4, 'A', 'C', 'B', 0.23806349924964598), (2, 3, 'A', 'C', 'B', 0.22623808445886484), (3, 2, 'A', 'C', 'B', 0.21645413617042697), (4, 1, 'A', 'C', 'B', 0.20805352247289927), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 5, 'A', 'B', 'C', 0)]
path [6] -> delay(30240000s): 467.4 -> 525.0
[(6, 0, 'C', 'A', 'B', 0.559487579332143), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (1, 5, 'A', 'C', 'B', 0.23806349924964598), (2, 4, 'A', 'C', 'B', 0.22623808445886484), (3, 3, 'A', 'C', 'B', 0.21645413617042697), (4, 2, 'A', 'C', 'B', 0.20805352247289927), (5, 1, 'A', 'C', 'B', 0.20508564098694648), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 6, 'A', 'B', 'C', 0)]

BIT_LEN = 6 critical path priority:
path [0] -> delay(30240000s): 338.2819 -> 404.0916
[(4, 0, 'C', 'A', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (4, 1, 'B', 'C', 'A', 0.27623934203576533), (4, 2, 'A', 'C', 'B', 0.13976290483496429), (4, 3, 'A', 'C', 'B', 0.08558649194731616), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (4, 4, 'C', 'A', 'B', 0.04410409370444246), (4, 5, 'B', 'A', 'C', 0.01562820271485621)]
path [1] -> delay(30240000s): 338.2819 -> 402.6629
[(4, 0, 'C', 'A', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (4, 1, 'B', 'C', 'A', 0.27623934203576533), (4, 2, 'A', 'C', 'B', 0.13976290483496429), (4, 3, 'A', 'C', 'B', 0.08558649194731616), (4, 4, 'C', 'A', 'B', 0.04410409370444246), (4, 5, 'B', 'A', 'C', 0.01562820271485621), (0, 1, 'A', 'B', 'C', 0)]
path [2] -> delay(30240000s): 338.0523 -> 395.6529
[(4, 0, 'C', 'A', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (4, 1, 'B', 'C', 'A', 0.27623934203576533), (1, 1, 'A', 'C', 'B', 0.23806349924964598), (4, 2, 'A', 'C', 'B', 0.13976290483496429), (4, 3, 'A', 'C', 'B', 0.08558649194731616), (4, 4, 'C', 'A', 'B', 0.04410409370444246), (4, 5, 'B', 'A', 'C', 0.01562820271485621), (0, 2, 'A', 'B', 'C', 0)]
path [3] -> delay(30240000s): 337.9679 -> 388.0931
[(4, 0, 'C', 'A', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (4, 1, 'B', 'C', 'A', 0.27623934203576533), (1, 2, 'A', 'C', 'B', 0.23806349924964598), (2, 1, 'A', 'C', 'B', 0.22623808445886484), (4, 2, 'A', 'C', 'B', 0.13976290483496429), (4, 3, 'A', 'C', 'B', 0.08558649194731616), (4, 4, 'C', 'A', 'B', 0.04410409370444246), (4, 5, 'B', 'A', 'C', 0.01562820271485621), (0, 3, 'A', 'B', 'C', 0)]
path [4] -> delay(30240000s): 337.8843 -> 380.0885
[(4, 0, 'C', 'A', 'B', 0.5548261609530758), (4, 1, 'B', 'C', 'A', 0.27623934203576533), (1, 3, 'A', 'C', 'B', 0.23806349924964598), (2, 2, 'A', 'C', 'B', 0.22623808445886484), (3, 1, 'A', 'C', 'B', 0.21645413617042697), (4, 2, 'A', 'C', 'B', 0.13976290483496429), (4, 3, 'A', 'C', 'B', 0.08558649194731616), (4, 4, 'C', 'A', 'B', 0.04410409370444246), (4, 5, 'B', 'A', 'C', 0.01562820271485621), (0, 4, 'A', 'B', 'C', 0)]


BIT_LEN = 10 critical path priority:


"""

from tool.log import Log
from tool import NBTI_formula as BTI
from msimulator.bin_func import signed_b, reverse_signed_b
from msimulator.Multiplier import Wallace_rew
from alpha import AlphaMultiprocess


from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body

from propagation_delay import wallace_multiplier_error_rate

import matplotlib.pyplot as plt
from datetime import datetime


BIT_LEN = 6
TEMP = 273.15 + 80
ALPHA_VERIFICATION = False
log = Log(f"{__file__}.{BIT_LEN}.log", terminal=True)


########################################################################################
################## Critical Path
########################################################################################

"""range(bit_len - 1)"""
def create_crit(i_th):
    crit = []
    for lay in range(0, BIT_LEN-2):
        crit += [
            (lay, max(i_th - lay, 0))
        ]
    crit += [(BIT_LEN-2, i) for i in range(BIT_LEN)]
    return crit

"""for multiplier propagation delay and optimization"""
CRITICAL_FA_lst = create_crit(0)

log.println(f"Critical eFA list: {CRITICAL_FA_lst}")

########################################################################################
##################### Functions
########################################################################################


def get_alpha(raw_mp, bit_len, log=log, rew_lst=[], verify=False):
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
        mp = Wallace_rew,
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
        plot = "DELAY" or "RATE" or True, 
        plot_save_clear = True,
        plot_label = "",
        alpha_verification = ALPHA_VERIFICATION, 
        critical_fa_lst = CRITICAL_FA_lst, 
        mp = Wallace_rew,
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
            log.println(f"week {week:03}: {delay: 8.3f} [{aging_rate * 100 :4.2f}%]")

        if plot:
            res_week.append(week)

            if plot == "DELAY":
                res_delay.append(delay)
            elif plot == "RATE" or True:
                res_delay.append(aging_rate)

    
    if plot:
        plt.plot(res_week, res_delay, label=plot_label)
        plt.title(f"WallceTree-BIT-{bit_len}-TEMP-{temp}")

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


if False and (__name__ == "__main__"):
    """each path gets full rewiring -> aging(1 year) -> route with higher aging higher priority"""
    log.println(f"RUNNING: critical path priorities, bit len [{BIT_LEN}]")
    for i in range(BIT_LEN - 1):
        critical_path = create_crit(i)
        _, worst_wiring = get_best_worst_wire_comb(
            log=False,
            bitlen=BIT_LEN,
            temp=TEMP,
            mp=Wallace_rew,
            critical_fa_lst=critical_path
        )

        ts_1year = 50 *7 *24*60*60

        alpha_nomitigation = get_alpha(Wallace_rew, BIT_LEN, log=False, rew_lst=[])
        alpha_rewired = get_alpha(Wallace_rew, BIT_LEN, log=False, rew_lst=worst_wiring)
        delay_nomitigation = get_MP_delay(critical_path, alpha_nomitigation, TEMP, ts_1year)
        delay_rewired = get_MP_delay(critical_path, alpha_rewired, TEMP, ts_1year)

        log.println(f"path [{i}] -> delay({ts_1year}s): {delay_nomitigation:.4f} -> {delay_rewired:.4f}")
        log.println(f"{sort_rewiring(worst_wiring)}")


"""specific wire combination aging"""
if False and (__name__ == "__main__"):
    # normal aging without mitigation

    REW_LST = []

    if False:
        if BIT_LEN == 8:
            """8-bit critical-path"""
            REW_LST = \
                [(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844)]
        elif BIT_LEN == 6:
            """6-bit critical-path"""
            REW_LST = \
                [(0, 0, 'C', 'A', 'B', 0.06047137087990345), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (4, 0, 'C', 'A', 'B', 0.5548261609530758), (4, 1, 'B', 'C', 'A', 0.27623934203576533), (4, 2, 'A', 'C', 'B', 0.13976290483496429), (4, 3, 'A', 'C', 'B', 0.08558649194731616), (4, 4, 'C', 'A', 'B', 0.04410409370444246), (4, 5, 'B', 'A', 'C', 0.01562820271485621)]
        elif BIT_LEN == 10:
            """10-bit critical-path"""
            REW_LST = \
                [(0, 0, 'C', 'A', 'B', 0.06047137087990345), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (6, 0, 'A', 'C', 'B', 0.559487579332143), (7, 0, 'A', 'C', 'B', 0.559487579332143), (8, 0, 'C', 'A', 'B', 0.559487579332143), (8, 1, 'A', 'C', 'B', 0.27974378966607144), (8, 2, 'A', 'C', 'B', 0.16752601088223223), (8, 3, 'A', 'C', 'B', 0.1318820895897786), (8, 4, 'A', 'C', 'B', 0.11401192182901981), (8, 5, 'A', 'C', 'B', 0.10202721396413394), (8, 6, 'C', 'A', 'B', 0.08961912187596943), (8, 7, 'C', 'A', 'B', 0.07844764707361834), (8, 8, 'C', 'A', 'B', 0.05196725706721206), (8, 9, 'A', 'B', 'C', 0)]
        else:
            raise RuntimeError("invalid BIT_LEN")

    #rewire top-10% of circuit
    circuit_size = BIT_LEN * (BIT_LEN - 1)
    REW_LST = sort_rewiring(REW_LST)
    REW_LST = REW_LST[:(circuit_size//10)]
    
    examine_wire_comb(
        wire_comb=REW_LST, 
        bit_len=BIT_LEN, 
        temp=TEMP, 
        log=log, 
        plot=True, 
        alpha_verification=ALPHA_VERIFICATION,
        critical_fa_lst=CRITICAL_FA_lst
        )

"""
extracting best and worst wiring combination for the provided multiplier
"""
if False and (__name__ == "__main__"):
    best_wiring, worst_wiring = get_best_worst_wire_comb(log=log)
    log.println(f"worst wiring:\n{worst_wiring}")
    
    # examine_multi_wire_comb(
    #     [worst_wiring, [], best_wiring],
    #     ["attack", "no-mitigation", "optimization"],
    #     log=log,
    #     plot="DELAY",
    # )


"""
partial rewiring
"""
if False and (__name__ == "__main__"):
    _, worst_wiring = get_best_worst_wire_comb(log=False)
    worst_wiring = sorted(worst_wiring, key=lambda x: x[-1], reverse=True)

    full_combo = []
    full_combo_label = []
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
if False:
    lst = []
    for fa_i in range(BIT_LEN - 1):
        for fa_j in range(BIT_LEN):
            lst += [(fa_i, fa_j)]

    best_wiring, worst_wiring = get_best_worst_wire_comb(critical_fa_lst=lst, log=False)
    log.println(f"worst complete wiring:\n{worst_wiring}")
    worst_wiring = sort_rewiring(worst_wiring)
    log.println(f"worst complete wiring (sorted):\n{worst_wiring}")



"""
error rate of wire combination
"""
if False and __name__ == "__main__":

    """6-bit M-1-100%"""
    # REW_LST = [(0, 0, 'C', 'A', 'B', 0.06047137087990345), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (4, 0, 'C', 'A', 'B', 0.5548261609530758), (4, 1, 'B', 'C', 'A', 0.27623934203576533), (4, 2, 'A', 'C', 'B', 0.13976290483496429), (4, 3, 'A', 'C', 'B', 0.08558649194731616), (4, 4, 'C', 'A', 'B', 0.04410409370444246), (4, 5, 'B', 'A', 'C', 0.01562820271485621)]
    """8-bit M-1-100%"""
    # REW_LST = [(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844)]
    """10-bit M-1-100%"""
    # REW_LST = [(0, 0, 'C', 'A', 'B', 0.06047137087990345), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (6, 0, 'A', 'C', 'B', 0.559487579332143), (7, 0, 'A', 'C', 'B', 0.559487579332143), (8, 0, 'C', 'A', 'B', 0.559487579332143), (8, 1, 'A', 'C', 'B', 0.27974378966607144), (8, 2, 'A', 'C', 'B', 0.16752601088223223), (8, 3, 'A', 'C', 'B', 0.1318820895897786), (8, 4, 'A', 'C', 'B', 0.11401192182901981), (8, 5, 'A', 'C', 'B', 0.10202721396413394), (8, 6, 'C', 'A', 'B', 0.08961912187596943), (8, 7, 'C', 'A', 'B', 0.07844764707361834), (8, 8, 'C', 'A', 'B', 0.05196725706721206), (8, 9, 'A', 'B', 'C', 0)]
    
    """6-bit M-all-100%"""
    # REW_LST = [(4, 0, 'C', 'A', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (4, 1, 'B', 'C', 'A', 0.27623934203576533), (1, 1, 'A', 'C', 'B', 0.23806349924964598), (1, 2, 'A', 'C', 'B', 0.23806349924964598), (1, 3, 'A', 'C', 'B', 0.23806349924964598), (2, 1, 'A', 'C', 'B', 0.22623808445886484), (2, 2, 'A', 'C', 'B', 0.22623808445886484), (3, 1, 'A', 'C', 'B', 0.21645413617042697), (3, 2, 'A', 'C', 'B', 0.1969030072855622), (3, 3, 'A', 'C', 'B', 0.1907827997015351), (2, 3, 'A', 'C', 'B', 0.18584052248128324), (3, 4, 'A', 'C', 'B', 0.1437575168356166), (4, 2, 'A', 'C', 'B', 0.13976290483496429), (2, 4, 'C', 'A', 'B', 0.11587346153684991), (1, 4, 'C', 'A', 'B', 0.09518042500764795), (4, 3, 'A', 'C', 'B', 0.08558649194731616), (1, 5, 'B', 'C', 'A', 0.07595053370645316), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (4, 4, 'C', 'A', 'B', 0.04410409370444246), (4, 5, 'B', 'A', 'C', 0.01562820271485621), (2, 5, 'C', 'B', 'A', 0.012848333528830136), (3, 5, 'C', 'B', 'A', 0.005668012802252154), (0, 1, 'A', 'B', 'C', 0), (0, 2, 'A', 'B', 'C', 0), (0, 3, 'A', 'B', 'C', 0), (0, 4, 'A', 'B', 'C', 0), (0, 5, 'A', 'B', 'C', 0)]
    """8-bit M-all-100%"""
    REW_LST = [(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (1, 1, 'A', 'C', 'B', 0.23806349924964598), (1, 2, 'A', 'C', 'B', 0.23806349924964598), (1, 3, 'A', 'C', 'B', 0.23806349924964598), (1, 4, 'A', 'C', 'B', 0.23806349924964598), (1, 5, 'A', 'C', 'B', 0.23806349924964598), (2, 1, 'A', 'C', 'B', 0.22623808445886484), (2, 2, 'A', 'C', 'B', 0.22623808445886484), (2, 3, 'A', 'C', 'B', 0.22623808445886484), (2, 4, 'A', 'C', 'B', 0.22623808445886484), (3, 1, 'A', 'C', 'B', 0.21645413617042697), (3, 2, 'A', 'C', 'B', 0.21645413617042697), (3, 3, 'A', 'C', 'B', 0.21645413617042697), (4, 1, 'A', 'C', 'B', 0.20805352247289927), (4, 2, 'A', 'C', 'B', 0.20805352247289927), (5, 1, 'A', 'C', 'B', 0.20508564098694648), (5, 2, 'A', 'C', 'B', 0.20101528375126826), (4, 3, 'A', 'C', 'B', 0.20044518222289298), (5, 3, 'A', 'C', 'B', 0.20040745491586842), (3, 4, 'A', 'C', 'B', 0.1969030072855622), (5, 4, 'A', 'C', 'B', 0.19681078497950177), (4, 4, 'A', 'C', 'B', 0.19510048039437622), (5, 5, 'A', 'C', 'B', 0.194203408871786), (4, 5, 'A', 'C', 'B', 0.19249310428666044), (3, 5, 'A', 'C', 'B', 0.1907827997015351), (2, 5, 'A', 'C', 'B', 0.18584052248128324), (5, 6, 'A', 'C', 'B', 0.17836743759193058), (4, 6, 'A', 'C', 'B', 0.16769824540818262), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (3, 6, 'A', 'C', 'B', 0.1437575168356166), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (2, 6, 'C', 'A', 'B', 0.11587346153684991), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (1, 6, 'C', 'A', 'B', 0.09518042500764795), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (1, 7, 'B', 'C', 'A', 0.07595053370645316), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (2, 7, 'C', 'B', 'A', 0.012848333528830136), (3, 7, 'C', 'B', 'A', 0.005668012802252154), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (4, 7, 'C', 'B', 'A', 0.0028235333545586494), (5, 7, 'C', 'B', 'A', 0.0028235333545586494), (0, 1, 'A', 'B', 'C', 0), (0, 2, 'A', 'B', 'C', 0), (0, 3, 'A', 'B', 'C', 0), (0, 4, 'A', 'B', 'C', 0), (0, 5, 'A', 'B', 'C', 0), (0, 6, 'A', 'B', 'C', 0), (0, 7, 'A', 'B', 'C', 0)]
    """10-bit M-all-100%"""
    # REW_LST = [(6, 0, 'A', 'C', 'B', 0.559487579332143), (7, 0, 'A', 'C', 'B', 0.559487579332143), (8, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (8, 1, 'A', 'C', 'B', 0.27974378966607144), (1, 1, 'A', 'C', 'B', 0.23806349924964598), (1, 2, 'A', 'C', 'B', 0.23806349924964598), (1, 3, 'A', 'C', 'B', 0.23806349924964598), (1, 4, 'A', 'C', 'B', 0.23806349924964598), (1, 5, 'A', 'C', 'B', 0.23806349924964598), (1, 6, 'A', 'C', 'B', 0.23806349924964598), (1, 7, 'A', 'C', 'B', 0.23806349924964598), (2, 1, 'A', 'C', 'B', 0.22623808445886484), (2, 2, 'A', 'C', 'B', 0.22623808445886484), (2, 3, 'A', 'C', 'B', 0.22623808445886484), (2, 4, 'A', 'C', 'B', 0.22623808445886484), (2, 5, 'A', 'C', 'B', 0.22623808445886484), (2, 6, 'A', 'C', 'B', 0.22623808445886484), (3, 1, 'A', 'C', 'B', 0.21645413617042697), (3, 2, 'A', 'C', 'B', 0.21645413617042697), (3, 3, 'A', 'C', 'B', 0.21645413617042697), (3, 4, 'A', 'C', 'B', 0.21645413617042697), (3, 5, 'A', 'C', 'B', 0.21645413617042697), (4, 1, 'A', 'C', 'B', 0.20805352247289927), (4, 2, 'A', 'C', 'B', 0.20805352247289927), (4, 3, 'A', 'C', 'B', 0.20805352247289927), (4, 4, 'A', 'C', 'B', 0.20805352247289927), (5, 1, 'A', 'C', 'B', 0.20508564098694648), (5, 2, 'A', 'C', 'B', 0.20508564098694648), (5, 3, 'A', 'C', 'B', 0.20508564098694648), (6, 1, 'A', 'C', 'B', 0.20220998180705446), (6, 2, 'A', 'C', 'B', 0.20220998180705446), (7, 1, 'A', 'C', 'B', 0.20220159796104858), (7, 2, 'A', 'C', 'B', 0.20220159796104858), (5, 4, 'A', 'C', 'B', 0.20101528375126826), (4, 5, 'A', 'C', 'B', 0.20044518222289298), (5, 5, 'A', 'C', 'B', 0.20040745491586842), (7, 5, 'A', 'C', 'B', 0.20038649530085484), (6, 3, 'A', 'C', 'B', 0.19987927261752092), (7, 3, 'A', 'C', 'B', 0.19987088877151504), (7, 4, 'A', 'C', 'B', 0.19926725185911814), (6, 5, 'A', 'C', 'B', 0.19866780686972374), (6, 4, 'A', 'C', 'B', 0.19754856342798738), (3, 6, 'A', 'C', 'B', 0.1969030072855622), (5, 6, 'A', 'C', 'B', 0.19681078497950177), (7, 6, 'A', 'C', 'B', 0.1967814415184821), (7, 7, 'A', 'C', 'B', 0.19592209730291676), (4, 6, 'A', 'C', 'B', 0.19510048039437622), (6, 6, 'A', 'C', 'B', 0.19506275308735133), (5, 7, 'A', 'C', 'B', 0.194203408871786), (6, 7, 'A', 'C', 'B', 0.194203408871786), (4, 7, 'A', 'C', 'B', 0.19249310428666044), (3, 7, 'A', 'C', 'B', 0.1907827997015351), (7, 8, 'A', 'C', 'B', 0.1891958714500634), (6, 8, 'A', 'C', 'B', 0.1873645921946439), (2, 7, 'A', 'C', 'B', 0.18584052248128324), (5, 8, 'A', 'C', 'B', 0.17836743759193058), (4, 8, 'A', 'C', 'B', 0.16769824540818262), (8, 2, 'A', 'C', 'B', 0.16752601088223223), (3, 8, 'A', 'C', 'B', 0.1437575168356166), (8, 3, 'A', 'C', 'B', 0.1318820895897786), (2, 8, 'C', 'A', 'B', 0.11587346153684991), (8, 4, 'A', 'C', 'B', 0.11401192182901981), (8, 5, 'A', 'C', 'B', 0.10202721396413394), (1, 8, 'C', 'A', 'B', 0.09518042500764795), (8, 6, 'C', 'A', 'B', 0.08961912187596943), (8, 7, 'C', 'A', 'B', 0.07844764707361834), (1, 9, 'B', 'C', 'A', 0.07595053370645316), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (8, 8, 'C', 'A', 'B', 0.05196725706721206), (2, 9, 'C', 'B', 'A', 0.012848333528830136), (3, 9, 'C', 'B', 'A', 0.005668012802252154), (4, 9, 'C', 'B', 'A', 0.0028235333545586494), (5, 9, 'C', 'B', 'A', 0.0028235333545586494), (0, 1, 'A', 'B', 'C', 0), (0, 2, 'A', 'B', 'C', 0), (0, 3, 'A', 'B', 'C', 0), (0, 4, 'A', 'B', 'C', 0), (0, 5, 'A', 'B', 'C', 0), (0, 6, 'A', 'B', 'C', 0), (0, 7, 'A', 'B', 'C', 0), (0, 8, 'A', 'B', 'C', 0), (0, 9, 'A', 'B', 'C', 0), (6, 9, 'A', 'B', 'C', 0), (7, 9, 'A', 'B', 'C', 0), (8, 9, 'A', 'B', 'C', 0)]

    log.println(f"RUNNING: error rate bitlen [{BIT_LEN}], REW_LST [{len(REW_LST)}]: \n{REW_LST}")

    alpha_notamper = get_alpha(Wallace_rew, BIT_LEN, log=False, rew_lst=[], verify=False)
    alpha = get_alpha(Wallace_rew, BIT_LEN, log=False, rew_lst=REW_LST, verify=False)
    log.println(f"alpha no-tamper: \n{alpha_notamper}")
    log.println(f"alpha REW [{len(REW_LST)}]: \n{alpha}")

    # fixed margin at 199 weeks aging
    margin_t_sec = 199 *7 *24*60*60
    max_ps_delay = get_MP_delay(CRITICAL_FA_lst, alpha_notamper, TEMP, margin_t_sec)
    log.println(f"max_ps_delay: {max_ps_delay}")
    
    res = []
    for t_week in range(200):
        t_sec = t_week *7 *24*60*60
        
        err_rate, max_seen_delay = wallace_multiplier_error_rate(BIT_LEN, alpha, TEMP, t_sec, max_ps_delay)
        res.append(err_rate)
        
        log.println(f"REW [{len(REW_LST)}] week [{t_week:03}], error rate: {err_rate:.4f}, max seen delay: {max_seen_delay:.3f}, max_allowed_delay: {max_ps_delay:.3f}")
    log.println(f"REW [{len(REW_LST)}], error rate: \n{res}")

