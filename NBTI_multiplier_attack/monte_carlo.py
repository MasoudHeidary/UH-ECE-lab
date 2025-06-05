


from tool.log import Log
import tool.NBTI_formula as BTI
import tool.life_expect as LX
from alpha import AlphaMultiprocess
from msimulator.Multiplier import MPn_rew, Wallace_rew

from mapping_tgate_pb_delay import tgate_pb_to_delay
from mapping_pmos_vth_body import pmos_vth_to_body

BIT_LEN = 8
TEMP = 273.15 + 30
log = Log(f"{__file__}.{BIT_LEN}.{TEMP}.log", terminal=True)


########################################################################################
################## Critical Path
########################################################################################

# array multiplier
CRITICAL_FA_lst = []
CRITICAL_FA_lst += [(0, i) for i in range(BIT_LEN)]
for lay in range(1, BIT_LEN-1):
    CRITICAL_FA_lst += [(lay, BIT_LEN-2), (lay, BIT_LEN-1)]
    

# REW_LST = []
REW_LST = [(0, 0, 'C', 'A', 'B', 0.018339334406103802), (0, 1, 'A', 'C', 'B', 0.07790750925369627), (0, 2, 'A', 'C', 'B', 0.07023986084793676), (0, 3, 'A', 'C', 'B', 0.06996676652115635), (0, 4, 'A', 'C', 'B', 0.06327805624062532), (0, 5, 'A', 'C', 'B', 0.06202602378861666), (0, 6, 'A', 'C', 'B', 0.06066055215471422), (0, 7, 'B', 'A', 'C', 0.031044522778167466), (1, 6, 'A', 'C', 'B', 0.026926601262683703), (1, 7, 'A', 'B', 'C', 0), (2, 6, 'A', 'C', 'B', 0.04209645826502892), (2, 7, 'A', 'C', 'B', 0.018389751820278716), (3, 6, 'A', 'C', 'B', 0.0520625793380578), (3, 7, 'A', 'C', 'B', 0.03514093767987467), (4, 6, 'A', 'C', 'B', 0.05473170075576064), (4, 7, 'A', 'C', 'B', 0.04195149004466142), (5, 6, 'A', 'C', 'B', 0.05910737854446707), (5, 7, 'A', 'C', 'B', 0.04896371206614772), (6, 6, 'C', 'A', 'B', 0.03757366354778771), (6, 7, 'B', 'A', 'C', 0.029334532147403697)]

AGED_TIME = 50*7 *24*60*60

########################################################################################
################## Functions
########################################################################################

def get_alpha(raw_mp, bit_len, rew_lst=[]):
    return AlphaMultiprocess(raw_mp, bit_len, rew_lst=rew_lst).run()


def monte_FA_delay(fa_pb):
    tg1_pb = max(fa_pb[0], fa_pb[1])
    tg2_pb = max(fa_pb[2], fa_pb[3])

    return tgate_pb_to_delay(tg1_pb) + tgate_pb_to_delay(tg2_pb)


def monte_MP_delay(crit_fa_lst, pb):
    ps = 0
    for fa_lay, fa_i in crit_fa_lst:
        ps += monte_FA_delay(pb[fa_lay][fa_i])
    return ps


def seed_generator(sample_index):
    return 7*sample_index + 1

########################################################################################
################## MAIN
########################################################################################


if True:
    alpha = get_alpha(MPn_rew, BIT_LEN, rew_lst=REW_LST)

    _min, _max = 999999, 0
    out = []
    
    for sample_index in range(0, 10_000):
        random_vth_matrix = LX.generate_guassian_vth_base(
            BIT_LEN, 
            seed=seed_generator(sample_index),
            sigma=0.1,
        )
        pb_matrix = LX.generate_body_from_base(
            BIT_LEN,
            random_vth_matrix,
            alpha,
            AGED_TIME,
            TEMP,
            pmos_map = pmos_vth_to_body
        )

        mp_ps = monte_MP_delay(CRITICAL_FA_lst, pb_matrix)
        mp_ps = int(mp_ps)
        # log.println(f"[{sample_index}]\t:{mp_ps}")
    
        if mp_ps < _min:
            _min = mp_ps
        if mp_ps > _max:
            _max = mp_ps
        out.append(mp_ps)
        
    log.println(f"min, max: {_min}, {_max}")
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    unique, counts = np.unique(out, return_counts=True)

    # Plot the frequency distribution
    plt.figure(figsize=(12, 6))
    plt.bar(unique, counts, width=1.0, color='skyblue', edgecolor='black')
    plt.title('Frequency Distribution of Normally Distributed Numbers (100-200)')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("monte.jpg")