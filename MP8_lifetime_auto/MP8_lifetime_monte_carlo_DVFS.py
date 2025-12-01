

"""

"""


from tool.log import Log

from msimulator.get_alpha_class import MultiplierStressTest
from msimulator.Multiplier import MPn_v3
from get_life_expect import *

import tool.NBTI_formula as NBTI
import tool.vth_body_map as VTH
from tool.map_pb_to_current import get_current_from_pb, get_pb_from_current

import itertools
import random
import multiprocessing



##############################
### optimizer conf
##############################

bit_len = 8
transistor_full_list = list(itertools.product(range(bit_len-1), range(bit_len), range(6)))


def random_optimizer_trigger(mp: MPn_v3, A, B):
    # return (A > 0)
    raise NotImplemented()



optimizer_random_bias = [0]
def random_optimizer_accept(neg_mp: MPn_v3, bin_A, bin_B):
    # global optimizer_random_bias
    # bias = optimizer_random_bias[0]

    # if random.random() < bias:
    #     return True
    # return False
    raise NotImplemented()



##############################
### monte carlo sampling computations
##############################

def seed_generator(i):
    return 7*i + 1
    
def get_monte_carlo_life_expect(alpha, vth_matrix, bit_len=bit_len):
    initial_v_base = VTH.get_body_voltage(vth=abs(NBTI.Vth))
    current_base = get_current_from_pb(initial_v_base)
    current_fail = current_base * (1 - 0.5)
    vb_fail = get_pb_from_current(current_fail)

    for t_week in range(200):
        t_sec = t_week * 7 * 24 * 60 * 60
        body_voltage = generate_body_voltage_from_base(
            alpha, t_sec, bit_len, vth_matrix
        )

        for fa_i in range(bit_len - 1):
            for fa_j in range(bit_len):
                for t_index in range(6):
                    if body_voltage[fa_i][fa_j][t_index] >= vb_fail:
                        return {
                            "fa_i": fa_i,
                            "fa_j": fa_j,
                            "t_index": t_index,
                            "t_week": t_week,
                        }


def monte_carlo_lifetime(
    bias,
    sample_count,
    new_alpha_step,
    log_obj = False
):
    sum_lifetime = 0
    alpha = None
    for sample_index in range(sample_count):
        
        if sample_index % new_alpha_step == 0:
            optimizer_random_bias[0] = bias
            alpha = MultiplierStressTest(bit_len, None, None, optimizer_enable=False).run(log_obj=False)
            if log_obj:
                log_obj.println("Generating new alpha")

        # random_vth_matrix = generate_random_vth_base()
        random_vth_matrix = generate_guassian_vth_base()
        fail_transistor = get_monte_carlo_life_expect(alpha, random_vth_matrix, bit_len)
        lifetime = fail_transistor["t_week"]
        sum_lifetime += lifetime

        if log_obj:
            log_obj.println(f"SAMPLE #{sample_index:03} (BIAS:{bias}) \tlifetime:{lifetime:3} weeks \\accumulated average {sum_lifetime/(sample_index+1)}")
            
    return sum_lifetime / sample_count



if True:
    """
        single run
    """
    BIAS = 1.0
    SAMPLE = 10_000
    NEW_ALPHA_STEP = 100
    DETAIL_LOG = True
    log = Log(f"{__file__}.log", terminal=True)

    lifetime = monte_carlo_lifetime(
        BIAS,
        SAMPLE,
        NEW_ALPHA_STEP,
        log_obj=log
    )        
    # final result
    log.println(f"final result [{SAMPLE}] samples, bias {BIAS}: \t {lifetime} weeks")
    log.println(f"\n")


