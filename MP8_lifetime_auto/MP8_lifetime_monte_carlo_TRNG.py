"""
RESULT:
    for TRNG with different bias values without considering is the number is positive or negative, no change in lifetime observed
"""


from tool.log import Log

from msimulator.get_alpha_class import MultiplierStressTest
from msimulator.Multiplier import MPn_v3
from get_life_expect import get_life_expect, generate_random_vth_base, generate_body_voltage_from_base
from sympy import symbols, Or, And, Not, simplify_logic
from pyeda.inter import expr

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
    return True



optimizer_random_bias = [0.5]
def random_optimizer_accept(neg_mp: MPn_v3, bin_A, bin_B):
    global optimizer_random_bias
    bias = optimizer_random_bias[0]

    if random.random() < bias:
        return True
    return False


##############################
### ideal computations (without variation)
##############################

if False:
    log = Log(f"{__file__}.log", terminal=True)


    for bias in range(0, 100+1, 20):

        log.println(f"bias {bias:3}%")
        log.println(f"Processing alpha")
        optimizer_random_bias[0] = bias/100
        alpha = MultiplierStressTest(bit_len, random_optimizer_trigger, random_optimizer_accept).run(log_obj=False)



        failed_transistor = get_life_expect(alpha, bit_len, faulty_transistor=False)
        log.println(f"faulty_transistor: \t{False} => failed_transistor: \t {failed_transistor}")
        log.println(f"\n")


##############################
### ideal computations (with variation)
##############################

if False:
    log = Log(f"{__file__}.log", terminal=True)
    DETAIL_LOG = False


    for bias in range(0, 100+1, 20):

        log.println(f"bias {bias:3}%")
        log.println(f"Processing alpha")
        optimizer_random_bias[0] = bias/100
        alpha = MultiplierStressTest(bit_len, random_optimizer_trigger, random_optimizer_accept).run(log_obj=False)


        sum_lifetime = 0

        for t in transistor_full_list:
            faulty_transistor = {'fa_i': t[0], 'fa_j': t[1], 't_index': t[2], 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
            failed_transistor = get_life_expect(alpha, bit_len, faulty_transistor)
            sum_lifetime += failed_transistor["t_week"]

            if DETAIL_LOG:
                log.println(f"faulty_transistor: \t{faulty_transistor} => failed_transistor: \t {failed_transistor}")

        len_transistor = len(transistor_full_list)
        log.println(f"final result >>> sum lifetime {sum_lifetime} / transistor len {len_transistor} = {sum_lifetime/len_transistor}")
        log.println(f"\n")






##############################
### monte carlo sampling computations
##############################

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


if True:
    log = Log(f"{__file__}.log", terminal=True)
    BIAS = 0.1
    SAMPLE = 100
    DETAIL_LOG = True

    sum_lifetime = 0
    for sample_index in range(SAMPLE):
        
        optimizer_random_bias[0] = BIAS/100
        alpha = MultiplierStressTest(bit_len, random_optimizer_trigger, random_optimizer_accept).run(log_obj=False)
    
        # find unoptimized failed transistor
        random_vth_matrix = generate_random_vth_base()
        fail_transistor = get_monte_carlo_life_expect(alpha, random_vth_matrix, bit_len)
        lifetime = fail_transistor["t_week"]
        sum_lifetime += lifetime

        if DETAIL_LOG:
            log.println(f"SAMPLE #{sample_index:03} (BIAS:{BIAS}) \tlifetime:{lifetime:3} weeks")

        
    # final result
    log.println(f"final result [{SAMPLE}] samples, bias {BIAS}: \t {sum_lifetime/SAMPLE} weeks")
    log.println(f"\n")

