

from tool.log import Log
from tool import NBTI_formula as BTI
from msimulator.Multiplier import MPn_rew
from alpha import AlphaMultiprocess

from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body

import random

BIT_LEN = 8
SAMPLE = 100_000
IN_ALPHA_SAMPLE = 1000
TEMP = 273.15 + 80
# AGE_TIME = (200-1) * 7 *24*60*60
AGE_TIME = (50) * 7 *24*60*60
# AGE_TIME = 100



########################################################################################
################## Rewiring List
########################################################################################

"""no-mitigation"""
REW_LST = []

"""half-critical path"""
# REW_LST = [(0, 1, 'A', 'C', 'B', 0.28976406320693054), (0, 2, 'A', 'C', 'B', 0.26259279119258666), (0, 3, 'A', 'C', 'B', 0.25481341220235604), (0, 4, 'A', 'C', 'B', 0.2315129781993061), (0, 5, 'A', 'C', 'B', 0.22822863079578365), (0, 6, 'A', 'C', 'B', 0.22620104898034382), (5, 6, 'A', 'C', 'B', 0.19532684423652524), (4, 6, 'A', 'C', 'B', 0.18361880328982083), (3, 6, 'A', 'C', 'B', 0.17180177234504546), (5, 7, 'A', 'C', 'B', 0.16337776734791393)]

"""one critical-path"""
# REW_LST = [(0, 0, 'C', 'A', 'B', 0.06047137087990345), (0, 1, 'A', 'C', 'B', 0.28976406320693054), (0, 2, 'A', 'C', 'B', 0.26259279119258666), (0, 3, 'A', 'C', 'B', 0.25481341220235604), (0, 4, 'A', 'C', 'B', 0.2315129781993061), (0, 5, 'A', 'C', 'B', 0.22822863079578365), (0, 6, 'A', 'C', 'B', 0.22620104898034382), (0, 7, 'B', 'A', 'C', 0.10020910946188893), (1, 6, 'A', 'C', 'B', 0.08965346783517958), (1, 7, 'A', 'B', 'C', 0), (2, 6, 'A', 'C', 'B', 0.1469926371678449), (2, 7, 'A', 'C', 'B', 0.06554451463712985), (3, 6, 'A', 'C', 'B', 0.17180177234504546), (3, 7, 'A', 'C', 'B', 0.11787204450625888), (4, 6, 'A', 'C', 'B', 0.18361880328982083), (4, 7, 'A', 'C', 'B', 0.14765893785803158), (5, 6, 'A', 'C', 'B', 0.19532684423652524), (5, 7, 'A', 'C', 'B', 0.16337776734791393), (6, 6, 'C', 'A', 'B', 0.1222909697594673), (6, 7, 'B', 'A', 'C', 0.09435069207269803)]

"""two ciritcal-path"""
# REW_LST = \
# list(
#     set(
#         [(0, 1, 'A', 'C', 'B', 0.28976406320693054), (0, 2, 'A', 'C', 'B', 0.26259279119258666), (0, 3, 'A', 'C', 'B', 0.25481341220235604), (0, 4, 'A', 'C', 'B', 0.2315129781993061), (0, 5, 'A', 'C', 'B', 0.22822863079578365), (0, 6, 'A', 'C', 'B', 0.22620104898034382), (5, 6, 'A', 'C', 'B', 0.19532684423652524), (4, 6, 'A', 'C', 'B', 0.18361880328982083), (3, 6, 'A', 'C', 'B', 0.17180177234504546), (5, 7, 'A', 'C', 'B', 0.16337776734791393), (4, 7, 'A', 'C', 'B', 0.14765893785803158), (2, 6, 'A', 'C', 'B', 0.1469926371678449), (6, 6, 'C', 'A', 'B', 0.1222909697594673), (3, 7, 'A', 'C', 'B', 0.11787204450625888), (0, 7, 'B', 'A', 'C', 0.10020910946188893), (6, 7, 'B', 'A', 'C', 0.09435069207269803), (1, 6, 'A', 'C', 'B', 0.08965346783517958), (2, 7, 'A', 'C', 'B', 0.06554451463712985), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (1, 7, 'A', 'B', 'C', 0)] +\
#         [(1, 1, 'A', 'C', 'B', 0.34500546869434406), (2, 1, 'A', 'C', 'B', 0.3426813216294845), (3, 1, 'A', 'C', 'B', 0.33285964603402163), (4, 1, 'A', 'C', 'B', 0.3313295941380153), (5, 1, 'A', 'C', 'B', 0.32992949185509374), (0, 1, 'A', 'C', 'B', 0.28976406320693054), (1, 2, 'A', 'C', 'B', 0.2835926597969251), (2, 2, 'A', 'C', 'B', 0.26321084534319295), (0, 2, 'A', 'C', 'B', 0.26259279119258666), (3, 2, 'A', 'C', 'B', 0.2559713943174293), (4, 2, 'A', 'C', 'B', 0.25051770249084093), (5, 2, 'A', 'C', 'B', 0.24158471457196307), (6, 4, 'C', 'A', 'B', 0.16147706598925166), (6, 3, 'C', 'A', 'B', 0.15606110146968832), (6, 5, 'C', 'A', 'B', 0.1495342773543939), (6, 2, 'C', 'A', 'B', 0.1327288580363356), (6, 6, 'C', 'A', 'B', 0.1222909697594673), (6, 1, 'B', 'A', 'C', 0.0979107455754254), (6, 7, 'B', 'A', 'C', 0.09435069207269803), (0, 0, 'C', 'A', 'B', 0.06047137087990345)]
#     )
# )


"""full circuit tampering"""
# REW_LST = \
# [(1, 1, 'A', 'C', 'B', 0.34500546869434406), (2, 1, 'A', 'C', 'B', 0.3426813216294845), (3, 1, 'A', 'C', 'B', 0.33285964603402163), (4, 1, 'A', 'C', 'B', 0.3313295941380153), (5, 1, 'A', 'C', 'B', 0.32992949185509374), (0, 1, 'A', 'C', 'B', 0.28976406320693054), (1, 2, 'A', 'C', 'B', 0.2835926597969251), (2, 2, 'A', 'C', 'B', 0.26321084534319295), (0, 2, 'A', 'C', 'B', 0.26259279119258666), (3, 2, 'A', 'C', 'B', 0.2559713943174293), (0, 3, 'A', 'C', 'B', 0.25481341220235604), (4, 2, 'A', 'C', 'B', 0.25051770249084093), (1, 3, 'A', 'C', 'B', 0.24528246539636517), (5, 2, 'A', 'C', 'B', 0.24158471457196307), (2, 3, 'A', 'C', 'B', 0.23865036847003196), (0, 4, 'A', 'C', 'B', 0.2315129781993061), (3, 3, 'A', 'C', 'B', 0.22857298557140127), (1, 4, 'A', 'C', 'B', 0.22826455908913784), (0, 5, 'A', 'C', 'B', 0.22822863079578365), (0, 6, 'A', 'C', 'B', 0.22620104898034382), (1, 5, 'A', 'C', 'B', 0.22271624390795836), (2, 4, 'A', 'C', 'B', 0.2206167157121659), (5, 3, 'A', 'C', 'B', 0.21565347887690034), (4, 3, 'A', 'C', 'B', 0.20973448359700508), (5, 4, 'A', 'C', 'B', 0.20495988329686377), (5, 5, 'A', 'C', 'B', 0.1982989176454809), (6, 0, 'C', 'A', 'B', 0.19815220034038428), (4, 4, 'A', 'C', 'B', 0.19672275459644367), (5, 0, 'C', 'A', 'B', 0.19582149115085073), (5, 6, 'A', 'C', 'B', 0.19532684423652524), (4, 0, 'C', 'A', 'B', 0.19347401426930622), (4, 5, 'A', 'C', 'B', 0.1912564870008468), (3, 4, 'A', 'C', 'B', 0.18889224242729097), (3, 0, 'C', 'A', 'B', 0.186381280548639), (4, 6, 'A', 'C', 'B', 0.18361880328982083), (3, 5, 'A', 'C', 'B', 0.17956521374615392), (2, 0, 'C', 'A', 'B', 0.17317253116694764), (3, 6, 'A', 'C', 'B', 0.17180177234504546), (5, 7, 'A', 'C', 'B', 0.16337776734791393), (6, 4, 'C', 'A', 'B', 0.16147706598925166), (6, 3, 'C', 'A', 'B', 0.15606110146968832), (2, 5, 'A', 'C', 'B', 0.1530815359278554), (6, 5, 'C', 'A', 'B', 0.1495342773543939), (4, 7, 'A', 'C', 'B', 0.14765893785803158), (2, 6, 'A', 'C', 'B', 0.1469926371678449), (1, 0, 'C', 'A', 'B', 0.14281882907082505), (6, 2, 'C', 'A', 'B', 0.1327288580363356), (6, 6, 'C', 'A', 'B', 0.1222909697594673), (3, 7, 'A', 'C', 'B', 0.11787204450625888), (0, 7, 'B', 'A', 'C', 0.10020910946188893), (6, 1, 'B', 'A', 'C', 0.0979107455754254), (6, 7, 'B', 'A', 'C', 0.09435069207269803), (1, 6, 'A', 'C', 'B', 0.08965346783517958), (2, 7, 'A', 'C', 'B', 0.06554451463712985), (0, 0, 'C', 'A', 'B', 0.06047137087990345), (1, 7, 'A', 'B', 'C', 0)]


log = Log(f"{__file__}.{BIT_LEN}.{TEMP}.log", terminal=True)

########################################################################################
################## Critical Path
########################################################################################

path_lst = []

"""auto, all critical path"""
def create_crit(i_th):
    crit = []
    for lay in range(1, BIT_LEN-2):
        crit += [(lay, i_th), (lay, i_th + 1)]
    crit += [(0, i) for i in range(0, i_th + 2)]
    crit += [(BIT_LEN-2, i) for i in range(i_th, BIT_LEN)]
    return crit

for i_th in range(BIT_LEN-1):
    path_lst += [create_crit(i_th)]

log.println(f"[{BIT_LEN}] critical path list: \n{path_lst}")


########################################################################################
################## Functions
########################################################################################

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


########################################################################################
################## MAIN
########################################################################################


if True and (__name__ == "__main__"):
    res = []

    alpha = get_alpha(MPn_rew, BIT_LEN, log=False, rew_lst=REW_LST)
    for sample_id in range(SAMPLE):
        delay = get_monte_MP_delay(sample_id, path_lst, alpha, TEMP, AGE_TIME)
        res.append(delay)

        if sample_id % 10_000 == 0:
            log.println(f"REW_LEN [{len(REW_LST)}]: sample [{sample_id:14,}] DONE")


    
    # delay from 0-1000 as counter, each delay increase by one
    _sample_per_delay = [0 for _ in range(1000)]
    for delay in res:
        _sample_per_delay[int(delay)] += 1
    
    log.println(f"REW LEN: {len(REW_LST)}")
    log.println(f"min: {min(res)}, max: {max(res)}")
    log.println(f"_sample_per_delay:\n{_sample_per_delay} \n")



if False and (__name__ == "__main__"):
    MIN_ALPHA = 0.1
    MAX_ALPHA = 0.9
    res = []

    for in_alpha_i in range(IN_ALPHA_SAMPLE):
        random.seed(in_alpha_i)
        a_alpha = [random.uniform(MIN_ALPHA, MAX_ALPHA) for _ in range(BIT_LEN)]
        b_alpha = [random.uniform(MIN_ALPHA, MAX_ALPHA) for _ in range(BIT_LEN)]
        in_alpha = {'A': a_alpha, 'B': b_alpha}
        
        alpha = get_monte_alpha(MPn_rew, BIT_LEN, 50_000, in_alpha, in_alpha_i, rew_lst=REW_LST)
        
        for sample_id in range(SAMPLE):
            delay = get_monte_MP_delay(sample_id, path_lst, alpha, TEMP, AGE_TIME)
            res.append(delay)
        log.println(f"[{BIT_LEN}] ALPHA[{in_alpha_i}] DONE")

    _sample_per_delay = [0 for _ in range(1000)]
    for delay in res:
        _sample_per_delay[int(delay)] += 1
    
    log.println(f"REW LEN: {len(REW_LST)}")
    log.println(f"{[BIT_LEN]} _sample_per_delay:\n{_sample_per_delay}")
