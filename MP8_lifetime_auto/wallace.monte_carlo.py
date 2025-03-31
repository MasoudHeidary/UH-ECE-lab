


from tool.log import Log

from msimulator.get_alpha_class import MultiplierStressTest
from msimulator.Multiplier import MPn_v3
from get_life_expect import *
from sympy import symbols, Or, And, Not, simplify_logic
from pyeda.inter import expr

import tool.NBTI_formula as NBTI
import tool.vth_body_map as VTH
from tool.map_pb_to_current import get_current_from_pb, get_pb_from_current

import itertools

from msimulator.Multiplier import Wallace_comp
from wallace_lifetime import wallace_alpha


import multiprocessing



##############################
### optimizer conf
##############################

optimizer_equation = [""]
def eq_optimizer_trigger(mp):
    global optimizer_equation
    
    eq = optimizer_equation[0]
    logical_expression = convert_logical_expression(eq)

    bin_A = mp.A
    bin_B = mp.B
    variables = {
        'B0': bin_B[0],
        'B1': bin_B[1],
        'B2': bin_B[2],
        'B3': bin_B[3],
        'B4': bin_B[4],
        'B5': bin_B[5],
        'B6': bin_B[6],
        'B7': bin_B[7],
        'A0': bin_A[0],
        'A1': bin_A[1],
        'A2': bin_A[2],
        'A3': bin_A[3],
        'A4': bin_A[4],
        'A5': bin_A[5],
        'A6': bin_A[6],
        'A7': bin_A[7],
    }

    result = eval(logical_expression, {}, variables)
    return result

def convert_logical_expression(expression):
    """Convert bitwise operators to Python logical operators."""
    return (expression
            .replace("&", "and")
            .replace("|", "or")
            .replace("~", "not "))

def eq_optimizer_accept(neg_mp):
    return True





##############################
### equation conf
##############################

bit_len = 8
full_list = list(itertools.product(range(bit_len-1), range(bit_len), range(6)))

optimizer_equation[0] = "(B0 & ~A6 & ~B1) | (B1 & ~A6 & ~B0) | (A6 & B6 & ~A7 & ~B7) | (A6 & B7 & ~A7 & ~B6)"


selector_conf = [
    ### no optimization
    [
        {
            'equation': '0',
            'transistor_list': [
                (fa_i, fa_j, t_index) for fa_i in range(bit_len-1) for fa_j in range(bit_len) for t_index in range(6)
            ],
            'alpha': wallace_alpha(Wallace_comp, 8, None, None, op_enable=False)
        }
    ],


    ### only healthy optimizer
    [
        {
            'equation': '(B0 & ~A6 & ~B1) | (B1 & ~A6 & ~B0) | (A6 & B6 & ~A7 & ~B7) | (A6 & B7 & ~A7 & ~B6)',
            'transistor_list': [
                (fa_i, fa_j, t_index) for fa_i in range(bit_len-1) for fa_j in range(bit_len) for t_index in range(6)
            ],
            'alpha': wallace_alpha(Wallace_comp, 8, eq_optimizer_trigger, eq_optimizer_accept, op_enable=True)
        }
    ],
]



##############################
### monte carlo sampling computations
##############################

def get_monte_carlo_life_expect(alpha, vth_matrix, bit_len=bit_len):
    initial_v_base = VTH.get_body_voltage(vth=abs(NBTI.Vth))
    current_base = get_current_from_pb(initial_v_base)
    current_fail = current_base * (1 - 0.5)
    vb_fail = get_pb_from_current(current_fail)

    for t_week in range(1000):
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
    raise ValueError("max lifetime is low")


if True:
    """multi process monte carlo real simulation"""

    PROCESS_POOL = 30
    log = Log(f"{__file__}.log", terminal=True)
    SAMPLE =  336 * 1000
    DETAIL_LOG = False
    CHART = True

    log.println(f"{selector_conf}")

    def seed_generator(i):
        return 7*i + 1

    def process_sample(sample_index, equation_conf, bit_len):
        """Function to process a single sample in parallel"""
        random_vth_matrix = generate_guassian_vth_base(seed=seed_generator(sample_index))

        optimize_equation = None
        max_optimizer_lifetime = 0
        max_failed_transistor = None

        for conf in equation_conf:
            alpha = conf['alpha']
            fail_transistor = get_monte_carlo_life_expect(alpha, random_vth_matrix, bit_len)
            lifetime = fail_transistor["t_week"]
            if lifetime > max_optimizer_lifetime:
                max_optimizer_lifetime = lifetime
                max_failed_transistor = fail_transistor
                optimize_equation = conf["equation"]

        optimized_fail_transistor = max_failed_transistor
        optimized_lifetime = max_optimizer_lifetime
        
        if DETAIL_LOG:
            log.println(f"{optimized_lifetime:03}")

        return optimized_lifetime, optimize_equation, optimized_fail_transistor


    for conf_index, conf_eq in enumerate(selector_conf):
        optimize_sum_lifetime = 0

        with multiprocessing.Pool(processes=PROCESS_POOL) as pool:
            results = pool.starmap(process_sample, [(i, conf_eq, bit_len) for i in range(SAMPLE)])

        for optimized_lifetime, optimize_equation, optimized_fail_transistor in results:
            optimize_sum_lifetime += optimized_lifetime


        if CHART:
            _sample_per_lifetime = [0 for _ in range(1000)]
            _min = 1000
            _max = 0

            for lifetime, _, _ in results:
                _sample_per_lifetime[lifetime] += 1

                if lifetime < _min:
                    _min = lifetime
                elif lifetime > _max:
                    _max = lifetime
            
            log.println(f"{_sample_per_lifetime}")
            log.println(f"min {_min}, max {_max}")


        # final result
        log.println(f"conf index: {conf_index}")
        log.println(f"final result [{SAMPLE}] samples: \t {optimize_sum_lifetime / SAMPLE} \n")
