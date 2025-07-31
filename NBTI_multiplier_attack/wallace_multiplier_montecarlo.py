

from tool.log import Log
from tool import NBTI_formula as BTI
from msimulator.Multiplier import Wallace_rew
from alpha import AlphaMultiprocess

from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body

import random

BIT_LEN = 8
SAMPLE = 300_000

IN_ALPHA_SAMPLE = 1000
SAMPLE = SAMPLE // IN_ALPHA_SAMPLE

TEMP = 273.15 + 80
# AGE_TIME = (200-1) * 7 *24*60*60      # 4 year
AGE_TIME = (50) * 7 *24*60*60           # 1 year
# AGE_TIME = 100                        # t=0 basically

########################################################################################
################## Rewiring List
########################################################################################
REW_LST = None
ALPHA_COUNTING_SAMPLE = None


if BIT_LEN == 8:
    ALPHA_COUNTING_SAMPLE = 40_000

    """no-mitigation"""
    # REW_LST = []

    """half critical-path"""
    # REW_LST =\
    # [(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (6, 1, 'B', 'C', 'A', 0.2797437896660715)]


    """one critical-path"""
    # REW_LST =\
    # [(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844)]

    """two critical-path"""
    # REW_LST = list(set(
    #     [(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844)]      +\
    #     [(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 1, 'A', 'B', 'C', 0)]
    # ))

    """full circuit tampering"""
    # REW_LST = list(set(
    #     [(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844)]      +\
    #     [(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (1, 0, 'A', 'C', 'B', 0.5237137084266037), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 1, 'A', 'B', 'C', 0)]                        +\
    #     [(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (2, 0, 'A', 'C', 'B', 0.5406406935117418), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (1, 1, 'A', 'C', 'B', 0.23806349924964598), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 2, 'A', 'B', 'C', 0)]                       +\
    #     [(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (3, 0, 'A', 'C', 'B', 0.5501312071899864), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (1, 2, 'A', 'C', 'B', 0.23806349924964598), (2, 1, 'A', 'C', 'B', 0.22623808445886484), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 3, 'A', 'B', 'C', 0)]                      +\
    #     [(6, 0, 'C', 'A', 'B', 0.559487579332143), (4, 0, 'A', 'C', 'B', 0.5548261609530758), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (1, 3, 'A', 'C', 'B', 0.23806349924964598), (2, 2, 'A', 'C', 'B', 0.22623808445886484), (3, 1, 'A', 'C', 'B', 0.21645413617042697), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 4, 'A', 'B', 'C', 0)]                     +\
    #     [(6, 0, 'C', 'A', 'B', 0.559487579332143), (5, 0, 'A', 'C', 'B', 0.5548261609530758), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (1, 4, 'A', 'C', 'B', 0.23806349924964598), (2, 3, 'A', 'C', 'B', 0.22623808445886484), (3, 2, 'A', 'C', 'B', 0.21645413617042697), (4, 1, 'A', 'C', 'B', 0.20805352247289927), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 5, 'A', 'B', 'C', 0)]                    +\
    #     [(6, 0, 'C', 'A', 'B', 0.559487579332143), (6, 1, 'B', 'C', 'A', 0.2797437896660715), (1, 5, 'A', 'C', 'B', 0.23806349924964598), (2, 4, 'A', 'C', 'B', 0.22623808445886484), (3, 3, 'A', 'C', 'B', 0.21645413617042697), (4, 2, 'A', 'C', 'B', 0.20805352247289927), (5, 1, 'A', 'C', 'B', 0.20508564098694648), (6, 2, 'A', 'C', 'B', 0.1596535794830523), (6, 3, 'A', 'C', 'B', 0.12236642437351741), (6, 4, 'A', 'C', 'B', 0.09636392598740745), (6, 5, 'C', 'A', 'B', 0.07807456592637296), (6, 6, 'C', 'A', 'B', 0.050249677430165784), (6, 7, 'B', 'A', 'C', 0.0034385392482901844), (0, 6, 'A', 'B', 'C', 0)]
    # ))

elif BIT_LEN == 6:
    pass

elif BIT_LEN == 10:
    pass

if REW_LST == None:
    raise RuntimeError("REW_LST is NONE, the defined Bit Length does not have corresponding rewiring")

log = Log(f"{__file__}.{BIT_LEN}.{TEMP}.log", terminal=True)
DUMP_RAW = True

########################## potential critical path list

path_lst = []

"""auto, all critical path"""
def create_crit(i_th):
    crit = []
    for lay in range(0, BIT_LEN-2):
        crit += [
            (lay, max(i_th - lay, 0))
        ]
    crit += [(BIT_LEN-2, i) for i in range(BIT_LEN)]
    return crit

for i_th in range(BIT_LEN-1):
    path_lst += [create_crit(i_th)]

log.println(f"[{BIT_LEN}] path list: \n{path_lst}")


########################## functions

def get_alpha(raw_mp, bit_len, log=False, rew_lst=[], verify=False):
    return AlphaMultiprocess(raw_mp, bit_len, log=log, rew_lst=rew_lst).run()

def get_monte_alpha(raw_mp, bit_len, sample_count, in_alpha, seed, log=False, rew_lst=[], verify=False):
    
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
    random.seed(seed)
    for alpha_sample_i in range(sample_count):
        a_bin = [
            random.choices([0,1], weights=[in_alpha['A'][bit], 1-in_alpha['A'][bit]], k=1)[0]
            for bit in range(bit_len)
        ]
        b_bin = [
            random.choices([0,1], weights=[in_alpha['B'][bit], 1-in_alpha['B'][bit]], k=1)[0]
            for bit in range(bit_len)
        ]

        mp = raw_mp(a_bin, b_bin, bit_len, rew_lst)
        mp.output

        #TODO: verify

        for row in range(alpha_row):
            for index in range(alpha_index):
                for t in range(6):
                    alpha[row][index][t] += (not mp.gfa[row][index].p[t])

    
    # alpha counter -> alpha probability
    for row in range(alpha_row):
        for index in range(alpha_index):
            for t in range(6):
                alpha[row][index][t] /= sample_count
    return alpha

def seed_generator(sample_index):
    return 7*sample_index + 1

def generate_guassian_vth_base(bit_len, mu=0, sigma=0.05, base_vth=abs(BTI.Vth), seed=False):
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

    return vth

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


########################## main

"""process variation monte carlo"""
if False and (__name__ == "__main__"):
    res = []

    log.println(f"RUNNING: monte carlo bit {BIT_LEN} with dynamic input alpha")
    log.println(f"config/ALPHA_COUTING_SAMPLE: {ALPHA_COUNTING_SAMPLE}")
    log.println(f"config/SAMPLE: {SAMPLE}")
    log.println(f"config/REW_LST: [{len(REW_LST)}] {REW_LST}")

    alpha = get_alpha(Wallace_rew, BIT_LEN, log=False, rew_lst=REW_LST)
    for sample_id in range(SAMPLE):
        delay = get_monte_MP_delay(sample_id, path_lst, alpha, TEMP, AGE_TIME)
        res.append(delay)

        if sample_id % 10_000 == 0:
            log.println(f"REW_LEN [{len(REW_LST)}]: sample [{sample_id:10,}/{SAMPLE:10,}] DONE")

    log.println(f"REW LEN: {len(REW_LST)}")
    log.println(f"min: {min(res)}, max: {max(res)}, average {sum(res)/len(res)}")
    if DUMP_RAW:
        log.println(f"raw result: \n{res}")
    
    # delay from 0-1000 as counter, each delay increase by one
    _sample_per_delay = [0 for _ in range(2000)]
    for delay in res:
        _sample_per_delay[int(delay)] += 1
    log.println(f"_sample_per_delay:\n{_sample_per_delay} \n")



"""process variation + dynamic input alpha"""
if True and (__name__ == "__main__"):
    MIN_ALPHA = 0.1
    MAX_ALPHA = 0.9
    res = []

    log.println("-"*20)
    log.println(f"RUNNING: monte carlo bit {BIT_LEN} with dynamic input alpha")
    log.println(f"config/ALPHA_COUTING_SAMPLE: {ALPHA_COUNTING_SAMPLE}")
    log.println(f"confing/IN_ALPHA_SAMPLE: {IN_ALPHA_SAMPLE}")
    log.println(f"config/SAMPLE: {SAMPLE}")
    log.println(f"config/REW_LST: [{len(REW_LST)}] {REW_LST}")
    log.println("-"*20)

    for in_alpha_i in range(IN_ALPHA_SAMPLE):
        random.seed(in_alpha_i)
        a_alpha = [random.uniform(MIN_ALPHA, MAX_ALPHA) for _ in range(BIT_LEN)]
        b_alpha = [random.uniform(MIN_ALPHA, MAX_ALPHA) for _ in range(BIT_LEN)]
        in_alpha = {'A': a_alpha, 'B': b_alpha}
        
        alpha = get_monte_alpha(Wallace_rew, BIT_LEN, ALPHA_COUNTING_SAMPLE, in_alpha, in_alpha_i, rew_lst=REW_LST)
        
        for sample_id in range(SAMPLE):
            delay = get_monte_MP_delay(sample_id, path_lst, alpha, TEMP, AGE_TIME)
            res.append(delay)
        log.println(f"- [{len(REW_LST)}] ALPHA[{in_alpha_i}] DONE")

    log.println(f"REW LEN: {len(REW_LST)}")
    log.println(f"[{len(REW_LST)}] min {min(res)}, max {max(res)}, average {sum(res)/len(res)}")
    if DUMP_RAW:
        log.println(f"raw result: \n{res}")

    _sample_per_delay = [0 for _ in range(2000)]
    for delay in res:
        _sample_per_delay[int(delay)] += 1
    log.println(f"[{len(REW_LST)}] _sample_per_delay:\n{_sample_per_delay}")
