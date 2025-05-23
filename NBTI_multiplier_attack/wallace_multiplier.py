

from tool.log import Log, Progress
from tool import NBTI_formula as BTI
from msimulator.bin_func import signed_b, reverse_signed_b
from msimulator.Multiplier import Wallace_rew

from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body

import matplotlib.pyplot as plt
from datetime import datetime

ALPHA_VERIFICATION = False

BIT_LEN = 6
TEMP = 273.15 + 30
log = Log(f"{__file__}.{BIT_LEN}.{TEMP}.log", terminal=True)


########################################################################################
################## Critical Path
########################################################################################

"""for multiplier propagation delay and optimization"""
CRITICAL_FA_lst = []

"""first FA in each row + all FA in last row"""
for lay in range(0, BIT_LEN-2):
    CRITICAL_FA_lst += [(lay, 0)]
CRITICAL_FA_lst += [(BIT_LEN-2, i) for i in range(BIT_LEN)]
log.println(f"Critical eFA list: {CRITICAL_FA_lst}")

"""second FA in each row + all FA in last row"""
# for lay in range(0, BIT_LEN-2):
#     CRITICAL_FA_lst += [(lay, 1)]
# CRITICAL_FA_lst += [(BIT_LEN-2, i) for i in range(BIT_LEN)]
# log.println(f"Critical eFA list: {CRITICAL_FA_lst}")

"""straight line"""
# for lay in range(0, BIT_LEN-2):
#     CRITICAL_FA_lst += [(lay, BIT_LEN - (lay + 2))]
# CRITICAL_FA_lst += [(BIT_LEN-2, i) for i in range(BIT_LEN)]
# log.println(f"Critical eFA list: {CRITICAL_FA_lst}")


########################################################################################
##################### Functions
########################################################################################


def get_alpha(raw_mp, bit_len, log=False, rew_lst=[], verify=False):
    bar = Progress(bars=1)

    alpha_row = bit_len-1
    alpha_index = bit_len
    alpha = [
        [
            [0 for _ in range(6)] 
            for _ in range(alpha_index)
        ]
        for _ in range(alpha_row)
    ]

    limit = 2 ** (bit_len - 1)
    for a in range(-limit, limit):
        bar.keep_line()
        bar.update(0, (a +limit)/(2*limit-1))

        for b in range(-limit, limit):

            a_bin = signed_b(a, bit_len)
            b_bin = signed_b(b, bit_len)

            mp: Wallace_rew
            mp = raw_mp(a_bin, b_bin, bit_len, rew_lst)
            mp.output
            
            if verify:
                out = reverse_signed_b(mp.output)
                if a * b != out:
                    raise ValueError(f"output verification failed, {a} * {b} != {out}")

            for row in range(alpha_row):
                for index in range(alpha_index):
                    for t in range(6):
                        alpha[row][index][t] += (not mp.gfa[row][index].p[t])

    # alpha couter -> alpha probability
    for row in range(alpha_row):
        for index in range(alpha_index):
            for t in range(6):
                alpha[row][index][t] /= ((2*limit)**2)

                # intercorrection, alpha 0 OR 1 -> 0.5
                if alpha[row][index][t] in [0, 1]:
                    alpha[row][index][t] = 0.5

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
        plot = True, 
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
            log.println(f"week {week:03}: {delay: 8.2f} [{aging_rate * 100 :4.2f}%]")

        res_week.append(week)
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


########################################################################################
########################################################################################
########################################################################################


"""specific wire combination aging"""
if False:
    # normal aging without mitigation
    examine_wire_comb(
        wire_comb=[], 
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
if True:
    # default wiring in multiplier
    best_wiring = [fa + ('A', 'B', 'C', 0) for fa in CRITICAL_FA_lst]
    worst_wiring = [fa + ('A', 'B', 'C', 0) for fa in CRITICAL_FA_lst]

    default_alpha = get_alpha(Wallace_rew, BIT_LEN, log=False, rew_lst=[], verify=ALPHA_VERIFICATION)
    log.println(f"{default_alpha}")
    log.println(f"default alpha done")


    # iterate in Critical FA list, and log the worst wiring for each
    for fa_index, fa in enumerate(CRITICAL_FA_lst):
        wire_combination = [
            ('A', 'B', 'C'),
            ('A', 'C', 'B'),
            ('B', 'A', 'C'),
            ('B', 'C', 'A'),
            ('C', 'A', 'B'),
            ('C', 'B', 'A'),
        ]
        lay, i = fa
        FA_zero_delay = get_FA_delay(default_alpha[lay][i], TEMP, 0)

        aging_period = 12*30 *24*60*60
        fa_default_delay = get_FA_delay(default_alpha[lay][i], TEMP, aging_period)
        fa_default_aging_rate = (fa_default_delay - FA_zero_delay) / FA_zero_delay
        log.println(f"{(lay, i)} default wiring, delay rate: {fa_default_aging_rate * 100 :.2f}% [t:{aging_period}s]")
        
        _worst_rate = fa_default_aging_rate
        _best_rate = fa_default_aging_rate
        for comb in wire_combination[1:]:
            rewire = fa + comb
            rewire_alpha = get_alpha(Wallace_rew, BIT_LEN, log=False, rew_lst=[rewire], verify=ALPHA_VERIFICATION)

            fa_delay = get_FA_delay(rewire_alpha[lay][i], TEMP, aging_period)
            fa_aging_rate = (fa_delay - FA_zero_delay) / FA_zero_delay
            log.println(f"{rewire}, delay rate: {fa_aging_rate * 100 :.2f}% [t:{aging_period}s]")

            if fa_aging_rate > _worst_rate:
                _worst_rate = fa_aging_rate
                worst_wiring[fa_index] = fa + comb + (fa_aging_rate - fa_default_aging_rate, )
                log.println(f"-new worst")
            if fa_aging_rate < _best_rate:
                _best_rate = fa_aging_rate
                best_wiring[fa_index] = fa + comb + (fa_aging_rate - fa_default_aging_rate, )
                log.println(f"-new best")
            
        log.println()
        
    log.println(f"best wiring combination: \n{best_wiring}")
    log.println(f"worst wiring combination: \n{worst_wiring}")
    

    examine_multi_wire_comb(
        [worst_wiring, [], best_wiring],
        ["attack", "no-mitigation", "optimization"],
        log=False,
        plot=True
    )


