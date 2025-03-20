


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

# cache
import pickle
from random import randint

import multiprocessing


##################### CACHE
class CACHE():
    def __init__(self, MAX_LENGTH=10000, filename=None):
        self.max_length = MAX_LENGTH
        self.filename = filename
        self.key_list = []
        self.value_list = []
        
    
    def hit_cache(self, key):
        return (key in self.key_list)
    
    def get_cache(self, key):
        if not self.hit_cache(key):
            raise LookupError("no cache hit")
        
        return self.value_list[self.key_list.index(key)]
    
    def add_cache(self, key, value=None):
        # overflow protection, remove a random value from cache
        if len(self.key_list) >= self.max_length:
            rnd_index = randint(0, self.max_length-1)
            self.key_list.pop(rnd_index)
            self.value_list.pop(rnd_index)

            # log.println(f"CACHE WARNING: overflow max [{self.max_length}]")

        self.key_list.append(key)
        self.value_list.append(value)

        if self.filename:
            self.dump_cache()

    def reset_cache(self):
        self.key_list = []
        self.value_list = []

    def dump_cache(self):
        if not self.filename:
            raise RuntimeError("in-memory cache can not be saved")
        
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)
        
    @classmethod
    def load_cache(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)



##############################
### optimizer conf
##############################

def eq_optimizer_trigger(mp: MPn_v3, A, B):
    return True

def convert_logical_expression(expression):
    """Convert bitwise operators to Python logical operators."""
    return (expression
            .replace("&", "and")
            .replace("|", "or")
            .replace("~", "not "))

optimizer_equation = [""]
def eq_optimizer_accept(neg_mp: MPn_v3, bin_A, bin_B):
    global optimizer_equation
    eq = optimizer_equation[0]
    logical_expression = convert_logical_expression(eq)

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






##############################
### equation conf
##############################

bit_len = 8
full_list = list(itertools.product(range(bit_len-1), range(bit_len), range(6)))

eq1_1_list = [
    (6, 7, 3), (1, 5, 1), (5, 6, 5), (0, 1, 0), (1, 5, 4), (5, 6, 2), (2, 1, 0), (0, 3, 0), (4, 2, 1), (4, 5, 0), (6, 6, 4), (2, 6, 2), (2, 7, 1), (5, 5, 0), (2, 6, 5), (1, 3, 0), (1, 1, 0), (3, 0, 0), (0, 6, 1), (0, 6, 4), (1, 5, 3), (3, 7, 5), (2, 4, 3), (3, 7, 2), (0, 3, 5), (6, 5, 4), (6, 6, 3), (4, 7, 2), (0, 3, 2), (4, 7, 5), (1, 3, 2), (5, 7, 2), (3, 3, 1), (1, 3, 5), (5, 7, 5), (0, 2, 0), (3, 6, 0), (2, 4, 1), (2, 4, 4), (6, 7, 4), (0, 7, 5), (0, 6, 3), (0, 7, 2), (2, 2, 0), (1, 7, 2), (1, 7, 5), (2, 7, 2), (1, 2, 0), (6, 5, 3), (2, 7, 5), (4, 6, 2), (0, 4, 5), (4, 6, 5), (3, 6, 2), (0, 2, 5), (3, 6, 5), (0, 2, 2), (0, 4, 2), 
]

eq2_1_list = [
    (6, 7, 3), (1, 5, 1), (5, 6, 5), (0, 1, 0), (1, 5, 4), (5, 6, 2), (2, 1, 0), (0, 3, 0), (4, 2, 1), (4, 5, 0), (6, 6, 4), (2, 6, 2), (2, 7, 1), (5, 5, 0), (2, 6, 5), (1, 3, 0), (0, 2, 5), (1, 1, 0), (3, 0, 0), (0, 4, 2), (0, 6, 1), (0, 6, 4), (1, 5, 3), (3, 7, 5), (3, 7, 2), (0, 3, 5), (6, 5, 4), (6, 6, 3), (4, 7, 2), (0, 3, 2), (4, 7, 5), (1, 3, 2), (5, 7, 2), (3, 3, 1), (1, 3, 5), (5, 7, 5), (0, 2, 0), (3, 6, 0), (2, 4, 1), (2, 4, 4), (6, 7, 4), (0, 6, 3), (2, 2, 0), (1, 7, 2), (1, 7, 5), (2, 7, 2), (1, 2, 0), (6, 5, 3), (2, 7, 5), (4, 6, 2), (4, 6, 5), (3, 6, 2), (2, 4, 3), (3, 6, 5), (0, 2, 2), (0, 4, 5), 
]
eq2_2_list = [
    (1, 6, 0), (1, 6, 2), (1, 6, 5), (2, 5, 0), (3, 4, 0), (0, 4, 0), (0, 7, 5), (0, 7, 2), (2, 5, 2), (2, 5, 5), 
]

eq3_1_list = [
    (6, 7, 3), (1, 5, 1), (5, 6, 5), (1, 5, 4), (5, 6, 2), (2, 1, 0), (0, 3, 0), (4, 2, 1), (4, 5, 0), (6, 6, 4), (2, 6, 2), (2, 7, 1), (5, 5, 0), (2, 6, 5), (1, 3, 0), (3, 6, 5), (3, 0, 0), (0, 6, 1), (0, 6, 4), (1, 5, 3), (3, 7, 5), (2, 4, 3), (3, 7, 2), (0, 3, 5), (6, 5, 4), (6, 6, 3), (4, 7, 2), (0, 3, 2), (4, 7, 5), (1, 3, 2), (5, 7, 2), (3, 3, 1), (1, 3, 5), (5, 7, 5), (0, 2, 0), (3, 6, 0), (2, 4, 1), (2, 4, 4), (6, 7, 4), (0, 6, 3), (2, 2, 0), (1, 7, 2), (1, 7, 5), (2, 7, 2), (1, 2, 0), (6, 5, 3), (2, 7, 5), (4, 6, 2), (4, 6, 5), (3, 6, 2), (0, 4, 2), (0, 4, 5), 
]
eq3_2_list = [
    (1, 6, 0), (1, 6, 2), (1, 6, 5), (2, 5, 0), (3, 4, 0), (0, 4, 0), (0, 7, 5), (0, 7, 2), (2, 5, 2), (2, 5, 5), 
]
eq3_3_list = [
    (0, 1, 0), (4, 6, 1), (1, 1, 0), (0, 1, 2), (0, 1, 5), (0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 2, 5), (0, 2, 2), 
]

eq4_1_list = [
    (6, 6, 4), (2, 6, 5), (0, 3, 5), (6, 5, 3), (4, 6, 2), (5, 6, 5), (0, 3, 0), (2, 6, 2), (5, 5, 0), (3, 6, 5), (3, 0, 0), (6, 5, 4), (6, 6, 3), (0, 3, 2), (0, 2, 0), (3, 6, 0), (1, 2, 0), (3, 6, 2), (5, 6, 2), (2, 1, 0), (4, 5, 0), (4, 6, 5), 
]
eq4_2_list = [
    (1, 6, 5), (3, 4, 0), (0, 7, 2), (1, 6, 0), (1, 6, 2), (2, 5, 5), (2, 5, 0), (0, 4, 0), (0, 7, 5), (2, 5, 2), 
]
eq4_3_list = [
    (1, 1, 0), (0, 0, 0), (0, 1, 5), (2, 0, 0), (0, 2, 5), (0, 1, 0), (4, 6, 1), (1, 0, 0), (0, 1, 2), (0, 2, 2), 
]
eq4_4_list = [
    (3, 5, 0), (1, 3, 0), (0, 6, 4), (1, 3, 2), (5, 7, 2), (3, 3, 4), (2, 2, 0), (2, 6, 0), (3, 1, 0), (6, 7, 3), (1, 5, 4), (2, 7, 1), (0, 6, 1), (3, 7, 5), (4, 7, 5), (3, 3, 1), (0, 6, 3), (2, 7, 5), (3, 3, 3), (1, 5, 1), (4, 7, 2), (2, 4, 4), (6, 7, 4), (1, 7, 5), (2, 7, 2), (0, 4, 5), (4, 2, 1), (1, 5, 3), (3, 7, 2), (1, 3, 5), (5, 7, 5), (2, 4, 1), (1, 7, 2), (2, 4, 3), (0, 4, 2), 
]


eq5_1_list = [
    (6, 6, 4), (2, 6, 5), (0, 3, 5), (6, 5, 3), (4, 6, 2), (5, 6, 5), (0, 3, 0), (2, 6, 2), (5, 5, 0), (3, 6, 5), (3, 0, 0), (6, 5, 4), (6, 6, 3), (0, 3, 2), (0, 2, 0), (3, 6, 0), (1, 2, 0), (3, 6, 2), (5, 6, 2), (2, 1, 0), (4, 5, 0), (4, 6, 5), 
]
eq5_2_list = [
    (1, 6, 5), (3, 4, 0), (0, 7, 2), (1, 6, 0), (1, 6, 2), (2, 5, 5), (2, 5, 0), (0, 7, 5), (2, 5, 2), (3, 4, 0), (0, 7, 2), (1, 6, 0), (1, 6, 2), (2, 5, 5), (2, 5, 0), (0, 7, 5), (2, 5, 2), (1, 6, 5), 
]
eq5_3_list = [
    (1, 1, 0), (0, 0, 0), (0, 1, 5), (2, 0, 0), (0, 2, 5), (0, 1, 0), (0, 1, 2), (0, 2, 2), (1, 0, 0), 
]
eq5_4_list = [
    (3, 5, 0), (0, 6, 4), (5, 7, 2), (3, 3, 4), (2, 6, 0), (6, 7, 3), (1, 5, 4), (2, 7, 1), (0, 6, 1), (3, 7, 5), (4, 7, 5), (3, 3, 1), (0, 6, 3), (2, 7, 5), (3, 3, 3), (1, 5, 1), (4, 7, 2), (2, 4, 4), (6, 7, 4), (1, 7, 5), (2, 7, 2), (0, 4, 5), (4, 2, 1), (1, 5, 3), (3, 7, 2), (5, 7, 5), (2, 4, 1), (1, 7, 2), (2, 4, 3), (0, 4, 2), 
]
eq5_5_list = [
    (1, 3, 0), (2, 4, 5), (1, 3, 2), (2, 2, 0), (3, 1, 0), (2, 4, 2), (4, 6, 1), (1, 3, 5), (0, 4, 0), (3, 3, 0), 
]

selector_conf = [
    ### no optimization
    [
        {
            'equation': '0',
            'transistor_list': [
                (fa_i, fa_j, t_index) for fa_i in range(bit_len-1) for fa_j in range(bit_len) for t_index in range(6)
            ],
            'alpha': None
        }
    ],


    ### only healthy optimizer
    [
        {
            'equation': '(B0 & ~A6) | (B0 & ~A7 & ~B1) | (A6 & B1 & ~A7 & ~B0)',
            'transistor_list': [
                (fa_i, fa_j, t_index) for fa_i in range(bit_len-1) for fa_j in range(bit_len) for t_index in range(6)
            ],
            'alpha': None
        }
    ],


    # +1 equation, cover 58 transistors
    [
        {
            'equation': '(~A4 & ~A7) | (~A2 & ~A3 & ~A7)',
            'transistor_list': eq1_1_list,
            'alpha': None
        },
        {
            'equation': '(B0 & ~A6) | (B0 & ~A7 & ~B1) | (A6 & B1 & ~A7 & ~B0)',
            'transistor_list': [t for t in full_list if t not in eq1_1_list],
            'alpha': None
        }
    ],


    # +2 equation, cover 66 transistors
    [
        {
            'equation': '(~A4 & ~A7) | (~A2 & ~A3 & ~A7)',
            'transistor_list': eq2_1_list,
            'alpha': None
        },
        {
            'equation': '(B0 & ~A5 & ~B1) | (B0 & ~A7 & ~B1) | (B1 & ~A7 & ~B0)',
            'transistor_list': eq2_2_list,
            'alpha': None
        },
        {
            'equation': '(B0 & ~A6) | (B0 & ~A7 & ~B1) | (A6 & B1 & ~A7 & ~B0)',
            'transistor_list': [t for t in full_list if t not in eq2_1_list+eq2_2_list],
            'alpha': None
        }
    ],


    # +3 equation, cover 72 transistors
    [
        {
            'equation': '(~A4 & ~A7) | (~A2 & ~A3 & ~A7)',
            'transistor_list': eq3_1_list,
            'alpha': None
        },
        {
            'equation': '(B0 & ~A5 & ~B1) | (B0 & ~A7 & ~B1) | (B1 & ~A7 & ~B0)',
            'transistor_list': eq3_2_list,
            'alpha': None
        },
        {
            'equation': '(A1 & A6 & ~A2) | (A0 & ~A1 & ~A2) | (A2 & A6 & ~A0 & ~A1)',
            'transistor_list': eq3_3_list,
            'alpha': None
        },
        {
            'equation': '(B0 & ~A6) | (B0 & ~A7 & ~B1) | (A6 & B1 & ~A7 & ~B0)',
            'transistor_list': [t for t in full_list if t not in eq3_1_list+eq3_2_list+eq3_3_list],
            'alpha': None
        }
    ],


    # +4 equation, cover 77 transistors
    [
        {
            'equation': '(~A4 & ~A7) | (~A2 & ~A3 & ~A7)',
            'transistor_list': eq4_1_list,
            'alpha': None
        },
        {
            'equation': '(B0 & ~A5 & ~B1) | (B0 & ~A7 & ~B1) | (B1 & ~A7 & ~B0)',
            'transistor_list': eq4_2_list,
            'alpha': None
        },
        {
            'equation': '(A1 & A6 & ~A2) | (A0 & ~A1 & ~A2) | (A2 & A6 & ~A0 & ~A1)',
            'transistor_list': eq4_3_list,
            'alpha': None
        },
        {
            'equation': '(~A4 & ~A7) | (~A5 & ~A7)',
            'transistor_list': eq4_4_list,
            'alpha': None
        },
        {
            'equation': '(B0 & ~A6) | (B0 & ~A7 & ~B1) | (A6 & B1 & ~A7 & ~B0)',
            'transistor_list': [t for t in full_list if t not in eq4_1_list + eq4_2_list + eq4_3_list + eq4_4_list],
            'alpha': None
        }
    ],


    # +5 equation, cover 80 transistors
    [
        {
            'equation': '(~A4 & ~A7) | (~A2 & ~A3 & ~A7)',
            'transistor_list': eq5_1_list,
            'alpha': None
        },
        {
            'equation': '(B0 & ~A5 & ~B1) | (B0 & ~A7 & ~B1) | (B1 & ~A7 & ~B0)',
            'transistor_list': eq5_2_list,
            'alpha': None
        },
        {
            'equation': '(A1 & A6 & ~A2) | (A0 & ~A1 & ~A2) | (A2 & A6 & ~A0 & ~A1)',
            'transistor_list': eq5_3_list,
            'alpha': None
        },
        {
            'equation': '(~A4 & ~A7) | (~A5 & ~A7)',
            'transistor_list': eq5_4_list,
            'alpha': None
        },
        {
            'equation': '(B0 & ~A5) | (B0 & ~B1) | (A6 & B1 & ~B0)',
            'transistor_list': eq5_5_list,
            'alpha': None
        },
        {
            'equation': '(B0 & ~A6) | (B0 & ~A7 & ~B1) | (A6 & B1 & ~A7 & ~B0)',
            'transistor_list': [t for t in full_list if t not in eq5_1_list + eq5_2_list + eq5_3_list + eq5_4_list + eq5_5_list],
            'alpha': None
        }
    ],
]


def preload_alpha(index):
    # loading alpha cache
    try:
        alpha_cache = CACHE.load_cache(f"{__file__}.alpha.cache")
    except:
        alpha_cache = CACHE(filename=f"{__file__}.alpha.cache")

    # base alpha
    if not alpha_cache.hit_cache('0'):
        optimizer_equation[0] = "0"
        base_alpha = MultiplierStressTest(bit_len, eq_optimizer_trigger, eq_optimizer_accept).run(log_obj=False)
        alpha_cache.add_cache('0', base_alpha)

    # preload equations alpha
    for conf in selector_conf[index]:
        equation = conf["equation"]
        if not alpha_cache.hit_cache(equation):
            optimizer_equation[0] = equation
            conf['alpha'] = MultiplierStressTest(bit_len, eq_optimizer_trigger, eq_optimizer_accept).run(log_obj=False)
            alpha_cache.add_cache(equation, conf['alpha'])
        else:
            conf['alpha'] = alpha_cache.get_cache(equation)

    return alpha_cache


##############################
### ideal computations
##############################

#old
if False:
    log = Log(f"{__file__}.log", terminal=True)
    DETAIL_LOG = False

    try:
        alpha_cache = CACHE.load_cache(f"{__file__}.alpha.cache")
    except:
        alpha_cache = CACHE(filename=f"{__file__}.alpha.cache")

    sum_lifetime = 0
    len_transistor = 0


    for conf_index, conf in enumerate(equation_conf):
        len_transistor += len(conf['transistor_list'])
        log.println(f"conf/{conf_index} += transistor len: {len_transistor}")

        equation = conf['equation']
        log.println(f"equation: {equation}")

        # calculate equation alpha
        alpha = None
        if not alpha_cache.hit_cache(equation):
            optimizer_equation[0] = equation
            alpha = MultiplierStressTest(bit_len, eq_optimizer_trigger, eq_optimizer_accept).run(log_obj=False)
            alpha_cache.add_cache(equation, alpha)
        else:
            alpha = alpha_cache.get_cache(equation)
        conf['alpha'] = alpha
        log.println(f"conf/{conf_index} alpha DONE")

        for t in conf['transistor_list']:
            faulty_transistor = {'fa_i': t[0], 'fa_j': t[1], 't_index': t[2], 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
            failed_transistor = get_life_expect(conf['alpha'], bit_len, faulty_transistor)
            sum_lifetime += failed_transistor["t_week"]

            if DETAIL_LOG:
                log.println(f"faulty_transistor: \t{faulty_transistor} => failed_transistor: \t {failed_transistor}")

    # final result
    log.println(f"conf:\n{equation_conf}")
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


#old
if False:
    log = Log(f"{__file__}.log", terminal=True)
    SAMPLE = 2
    DETAIL_LOG = True

    alpha_cache = preload_alpha()
    log.println(f"preload_alpha DONE + equations alphas")

    base_sum_lifetime = 0
    optimize_sum_lifetime = 0

    # base alpha
    base_alpha = alpha_cache.get_cache("0")

    for sample_index in range(SAMPLE):
        
        # find unoptimized failed transistor
        random_vth_matrix = generate_random_vth_base()
        base_fail_transistor = get_monte_carlo_life_expect(base_alpha, random_vth_matrix, bit_len)
        base_lifetime = base_fail_transistor["t_week"]
        base_sum_lifetime += base_lifetime

        if False:
            # choose optimizer based on unoptimized failed transistor
            optimize_equation = None
            optimize_alpha = None
            for conf in equation_conf:
                if (base_fail_transistor['fa_i'], base_fail_transistor['fa_j'], base_fail_transistor['t_index']) in conf['transistor_list']:
                    optimize_equation = conf['equation']
                    optimize_alpha = conf['alpha']
                    break
        
            # failed transistor after optimization
            optimized_fail_transistor = get_monte_carlo_life_expect(optimize_alpha, random_vth_matrix, bit_len)
            optimized_lifetime = optimized_fail_transistor['t_week']
            optimize_sum_lifetime += optimized_lifetime
        
        else:
            # choose optimizer based on which one gives higher lifetime
            optimize_equation = None
            optimize_alpha = None
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
                    optimize_alpha = alpha
            
            optimized_fail_transistor = max_failed_transistor
            optimized_lifetime = max_optimizer_lifetime
            optimize_sum_lifetime += optimized_lifetime


        if DETAIL_LOG:
            log.println(f"[{base_lifetime:3d}] -> [{optimized_lifetime:3d}] \t|| {base_fail_transistor} => {optimize_equation} => {optimized_fail_transistor}")

        
    # final result
    log.println(f"conf:\n{equation_conf}\n")
    log.println(f"final result [{SAMPLE}] samples: \t {base_sum_lifetime/SAMPLE} => {optimize_sum_lifetime/SAMPLE}")



# multi process monte carlo real simulation
if True:

    PROCESS_POOL = 35
    log = Log(f"{__file__}.log", terminal=True)
    SAMPLE =  336 * 1000  # len(transistors) * samples
    DETAIL_LOG = False

    def seed_generator(i):
        return 7*i + 1

    def process_sample(sample_index, base_alpha, equation_conf, bit_len):
        """Function to process a single sample in parallel"""
        # random_vth_matrix = generate_random_vth_base()
        random_vth_matrix = generate_guassian_vth_base(seed=seed_generator(sample_index))
        
        # base_fail_transistor = get_monte_carlo_life_expect(base_alpha, random_vth_matrix, bit_len)
        # base_lifetime = base_fail_transistor["t_week"]

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

        return optimized_lifetime, optimize_equation, optimized_fail_transistor


    for conf_index, conf_eq in enumerate(selector_conf):
        alpha_cache = preload_alpha(conf_index)
        base_alpha = alpha_cache.get_cache("0")
        log.println(f"preload_alpha DONE + equations alphas")


        optimize_sum_lifetime = multiprocessing.Value('d', 0.0)
        lock = multiprocessing.Lock()

        with multiprocessing.Pool(processes=PROCESS_POOL) as pool:
            results = pool.starmap(process_sample, [(i, base_alpha, conf_eq, bit_len) for i in range(SAMPLE)])

        for optimized_lifetime, optimize_equation, optimized_fail_transistor in results:
            with lock:
                optimize_sum_lifetime.value += optimized_lifetime

            if DETAIL_LOG:
                log.println(f"[{optimized_lifetime:3d}] \t||{optimize_equation} => {optimized_fail_transistor}")

        # final result
        log.println(f"conf index: {conf_index}")
        log.println(f"final result [{SAMPLE}] samples: \t {optimize_sum_lifetime.value / SAMPLE} \n")
