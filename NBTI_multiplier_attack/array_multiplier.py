

from tool.log import Log, Progress
from tool import NBTI_formula as BTI
from msimulator.bin_func import signed_b
from msimulator.Multiplier import MPn_rew

from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body

BIT_LEN = 8
TEMP = 273.15 + 80
log = Log(f"{__file__}.{BIT_LEN}.{TEMP}.log", terminal=True)

"""for multiplier propagation delay and optimization"""
CRITICAL_FA_lst = []

"""all FA in first row + last 2 FA in each row """
# CRITICAL_FA_lst += [(0, i) for i in range(BIT_LEN)]
# for lay in range(1, BIT_LEN-1):
#     CRITICAL_FA_lst += [(lay, BIT_LEN-2), (lay, BIT_LEN-1)]
# log.println(f"Critical eFA list: {CRITICAL_FA_lst}")

"""first 2 FA in each row + all FA in last row"""
for lay in range(0, BIT_LEN-2):
    CRITICAL_FA_lst += [(lay, 0), (lay, 1)]
CRITICAL_FA_lst += [(BIT_LEN-2, i) for i in range(BIT_LEN)]
log.println(f"Critical eFA list: {CRITICAL_FA_lst}")


def get_alpha(raw_mp, bit_len, log=False, rew_lst=[]):
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

            mp: MPn_rew
            mp = raw_mp(a_bin, b_bin, bit_len, rew_lst)
            mp.output

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


# examine how unbalance the alpha in FA (Tgate0, Tgate1) is
# def get_unbalance_score(alpha, lay, i):
#     alp = alpha[lay][i]
    
#     return abs(alp[0] - alp[1]) + abs(alp[2] - alp[3])


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
extracting best and worst wiring combination for the provided multiplier
"""
if False:
    # default wiring in multiplier
    best_wiring = [fa + ('A', 'B', 'C', 0) for fa in CRITICAL_FA_lst]
    worst_wiring = [fa + ('A', 'B', 'C', 0) for fa in CRITICAL_FA_lst]

    default_alpha = get_alpha(MPn_rew, BIT_LEN, log=False, rew_lst=[])
    # log.println(f"{default_alpha}")
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
        log.println(f"default wiring, delay rate: {fa_default_aging_rate * 100 :.2f}% [t:{aging_period}s]")
        
        _worst_rate = fa_default_aging_rate
        _best_rate = fa_default_aging_rate
        for comb in wire_combination[1:]:
            rewire = fa + comb
            rewire_alpha = get_alpha(MPn_rew, BIT_LEN, log=False, rew_lst=[rewire])

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
    
    

            

"""
wire combination notation:

(0, 0, 'C', 'B', 'A', 0.50)
- FA index
- FA wiring combination
- difference of aging_rate and default_aging_rate
"""

if True:    
    wire_comb = []
        
    
    log.println(f"aging log for following wire comb \n{wire_comb}")
    
    alpha = get_alpha(MPn_rew, BIT_LEN, log=False, rew_lst=wire_comb)
    MP_zero_delay = get_MP_delay(CRITICAL_FA_lst, alpha, TEMP, 0)
    
    for week in range(0, 200):
        delay = get_MP_delay(CRITICAL_FA_lst, alpha, TEMP, week * 7 *24*60*60)
        aging_rate = (delay - MP_zero_delay) / MP_zero_delay
        log.println(f"week {week:03}: {delay: 8.2f} [{aging_rate * 100 :4.2f}%]")
        