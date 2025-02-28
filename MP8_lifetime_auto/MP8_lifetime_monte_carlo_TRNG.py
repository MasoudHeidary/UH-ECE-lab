"""
RESULT:
    for TRNG with different bias values without considering is the number is positive or negative, no change in lifetime observed

activation function: (A > 0) and (B > 0)
final result [5000] samples, bias 0.0: 	 57.3712 weeks
final result [5000] samples, bias 0.1: 	 59.284 weeks
final result [5000] samples, bias 0.2: 	 61.7998 weeks
final result [5000] samples, bias 0.3: 	 63.5816 weeks
final result [5000] samples, bias 0.4: 	 65.094 weeks
final result [5000] samples, bias 0.5: 	 66.3648 weeks
final result [5000] samples, bias 0.6: 	 67.8632 weeks
final result [5000] samples, bias 0.7: 	 68.9912 weeks
final result [5000] samples, bias 0.8: 	 69.6658 weeks
final result [5000] samples, bias 1.0: 	 70.5918 weeks
final result [5000] samples, bias 0.9: 	 70.3436 weeks


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
    # return True
    return (A > 0) and (B > 0)
    # return (A > 0)



optimizer_random_bias = [0.1]
def random_optimizer_accept(neg_mp: MPn_v3, bin_A, bin_B):
    global optimizer_random_bias
    bias = optimizer_random_bias[0]

    if random.random() < bias:
        return True
    return False



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
            alpha = MultiplierStressTest(bit_len, random_optimizer_trigger, random_optimizer_accept).run(log_obj=False)
            if log_obj:
                log_obj.println("Generating new alpha")

        random_vth_matrix = generate_random_vth_base()
        fail_transistor = get_monte_carlo_life_expect(alpha, random_vth_matrix, bit_len)
        lifetime = fail_transistor["t_week"]
        sum_lifetime += lifetime

        if log_obj:
            log_obj.println(f"SAMPLE #{sample_index:03} (BIAS:{bias}) \tlifetime:{lifetime:3} weeks \\accumulated average {sum_lifetime/(sample_index+1)}")
            
    return sum_lifetime / sample_count



if False:
    """
        single run
    """
    BIAS = 1.0
    SAMPLE = 10_000
    NEW_ALPHA_STEP = 200
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


if True:
    """
        Auto run for a range of bias using individual processes
    """
    SAMPLE = 5_000
    NEW_ALPHA_STEP = 200
    DETAIL_LOG = True
    log = Log(f"{__file__}.log", terminal=True)


    def run_simulation(bias):
        lifetime = monte_carlo_lifetime(
            bias,
            SAMPLE,
            NEW_ALPHA_STEP,
            log_obj=log
        )
        log.println(f"final result [{SAMPLE}] samples, bias {bias}: \t {lifetime} weeks")
        log.println(f"\n")



    processes = []
    for bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p = multiprocessing.Process(target=run_simulation, args=(bias,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
