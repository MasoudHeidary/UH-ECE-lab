


from tool.log import Log

from msimulator.get_alpha_class import MultiplierStressTest
from msimulator.Multiplier import *
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

def normalize_dict_list(dict_list):
    """
    Converts list of dicts into a hashable, order-independent representation.
    """
    normalized = tuple(
        sorted(
            frozenset((k, v) for k, v in sorted(d.items()))
            for d in dict_list
        )
    )
    return normalized


##############################
### optimizer
##############################
lst_transistor_optimize = []

def optimizer_trigger(mp: MPn_v3, _a, _b):
    for transistor in lst_transistor_optimize:
        fa_i:int = transistor['fa_i']
        fa_j:int = transistor['fa_j']

        t_index:int = transistor['t_index']
            
        if mp.gfa[fa_i][fa_j].p[t_index] == L:
            return True

    return False

def optimizer_accept(neg_mp: MPn_v3, _a, _b):
    for transistor in lst_transistor_optimize:
        fa_i:int = transistor['fa_i']
        fa_j:int = transistor['fa_j']

        t_index:int = transistor['t_index']

        if neg_mp.gfa[fa_i][fa_j].p[t_index] == H:
            return True
    return False

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
    # cache_name = f"{__file__}.alpha.cache"
    cache_name = f"MP8_lifetime_monte_carlo.py.alpha.cache"
    try:
        alpha_cache = CACHE.load_cache(filename=cache_name)
    except:
        alpha_cache = CACHE(filename=cache_name)

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
### monte carlo sampling computations
##############################

def get_monte_carlo_life_expect(alpha, vth_matrix_t0, t0_week, bit_len=bit_len):
    initial_v_base = VTH.get_body_voltage(vth=abs(NBTI.Vth))
    current_base = get_current_from_pb(initial_v_base)
    current_fail = current_base * (1 - 0.5)
    vb_fail = get_pb_from_current(current_fail)

    for t_week in range(1000):
        t_sec = t_week * 7 * 24 * 60 * 60
        t0_sec = t0_week * 7 * 24 * 60 * 60
        # body_voltage = generate_body_voltage_from_base(
        #     alpha, t_sec, bit_len, vth_matrix
        # )
        vth_aged_matrix = BTI_aging_step(alpha, vth_matrix_t0, t0_sec, t0_sec + t_sec)
        body_voltage = BTI_body_voltage(vth_aged_matrix)

        for fa_i in range(bit_len - 1):
            for fa_j in range(bit_len):
                for t_index in range(6):
                    if body_voltage[fa_i][fa_j][t_index] >= vb_fail:
                        return {
                            "fa_i": fa_i,
                            "fa_j": fa_j,
                            "t_index": t_index,
                            "t_week": t_week + t0_week,
                        }
    # return {
    #     "fa_i": 0,
    #     "fa_j": 0,
    #     "t_index": 0,
    #     "t_week": 1000,
    # }
    raise ValueError("max lifetime is low")


def BTI_aging_step(alpha, vth_matrix_t0, t0, t1):
    vth_matrix = [
        [[0 for _ in range(6)] for _ in range(bit_len)] for _ in range(bit_len-1)
    ]
    
    for fa_i in range(bit_len-1):
        for fa_j in range(bit_len):
            for t_index in range(6):
                vth_1 = NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha[fa_i][fa_j][t_index], NBTI.Tclk, t1)
                vth_0 = NBTI.delta_vth(NBTI.Vdef, NBTI.T, alpha[fa_i][fa_j][t_index], NBTI.Tclk, t0)
                vth_growth = vth_1 - vth_0

                vth_matrix[fa_i][fa_j][t_index] = vth_matrix_t0[fa_i][fa_j][t_index] + vth_growth

    return vth_matrix

def BTI_body_voltage(vth_matrix):
    body_voltage = [
        [[0 for _ in range(6)] for _ in range(bit_len)] for _ in range(bit_len-1)
    ]

    for fa_i in range(bit_len-1):
        for fa_j in range(bit_len):
            for t_index in range(6):
                body_voltage[fa_i][fa_j][t_index] = VTH.get_body_voltage(vth_matrix[fa_i][fa_j][t_index])
                
    return body_voltage


def get_special_SM_alpha(vth_matrix_t0, t0, log=False, bit_len=bit_len):
    global lst_transistor_optimize
    lst_transistor_optimize = []

    l_cache_name = "tmp.cache"
    try:
        l_cache = CACHE.load_cache(filename=l_cache_name)
    except:
        l_cache = CACHE(filename=l_cache_name)

    for i in range(10):
        if len(lst_transistor_optimize) != i:
            break
        
        if l_cache.hit_cache(normalize_dict_list(lst_transistor_optimize)):
            alpha = l_cache.get_cache(normalize_dict_list(lst_transistor_optimize))
        else:
            alpha = MultiplierStressTest(bit_len, optimizer_trigger, optimizer_accept, True).run()
            l_cache.add_cache(normalize_dict_list(lst_transistor_optimize), alpha)

        fail_transistor = get_monte_carlo_life_expect(alpha, vth_matrix_t0, t0, bit_len)
        fail_transistor.pop("t_week")

        if fail_transistor in lst_transistor_optimize:
            break

        lst_transistor_optimize.append(fail_transistor)
        lst_transistor_optimize = list(lst_transistor_optimize)

    return alpha

# multi process monte carlo real simulation
if False:

    PROCESS_POOL = 40
    log = Log(f"{__file__}.log", terminal=True)
    # SAMPLE =  336 * 1000  # len(transistors) * samples
    SAMPLE =  5000  # len(transistors) * samples
    DETAIL_LOG = False
    CHART = True

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

        """to check if the event caused the circuit to change the failed transistor location"""
        original_failed_transistor = None
        
        vth_matrix_t0 = random_vth_matrix
        # for event_time in [0, 25, 50, 75, 100, 125, 150, 175, 200, 1000]:
        for event_time in [0, 1000]:
            event_time_next = event_time + 1000
            
            for conf in equation_conf:
                alpha = conf['alpha']
                fail_transistor = get_monte_carlo_life_expect(alpha, vth_matrix_t0, event_time, bit_len)
                lifetime = fail_transistor["t_week"]
                
                if lifetime > max_optimizer_lifetime:
                    max_optimizer_lifetime = lifetime
                    max_failed_transistor = fail_transistor
                    optimize_equation = conf["equation"]

            if event_time == 0:
                original_failed_transistor = max_failed_transistor
                    
            if max_optimizer_lifetime < event_time_next:
                """if the max lifetime is bigger it means that before that failure an event will happens"""
                special_alpha = get_special_SM_alpha(vth_matrix_t0, False, bit_len)
                special_lifetime = get_monte_carlo_life_expect(special_alpha, vth_matrix_t0, event_time, bit_len)

                """compare the lifetime by using regular SM0, and specialized SM"""
                log.println(f"[{sample_index}]: SM0: {lifetime:3}, special SM: {special_lifetime}")

                break

            if event_time != 0:
                """restart the max lifetime numbers"""
                max_optimizer_lifetime = 0
                max_failed_transistor = None
                
                """apply the event, update the Vth"""
                event_time_sec = event_time * 7 * 24 * 60 * 60
                event_time_next_sec = event_time_next * 7 * 24 * 60 * 60
                vth_matrix_t0 = BTI_aging_step(alpha, vth_matrix_t0, event_time_sec, event_time_next_sec)
                
                """a sudden 2% increase in three transistor"""
                random.seed(seed_generator(sample_index))
                for t in range(3):
                    r_fa_i = random.randint(0, bit_len-2)
                    r_fa_j = random.randint(0, bit_len-1)
                    r_t_index = random.randint(0, 5)
                    vth_matrix_t0[r_fa_i][r_fa_j][r_t_index] *= 1.02
                
                # log.println(f"EVENT")
                
        optimized_fail_transistor = max_failed_transistor
        optimized_lifetime = max_optimizer_lifetime

        # oft1 = original_failed_transistor.copy()
        # oft1.pop('t_week')
        # oft2 = optimized_fail_transistor.copy()
        # oft2.pop('t_week')
        # _change_flag = not (oft1 == oft2)
        # # log.println(f"event changed transistor [{not _change_flag}]")
        
        if DETAIL_LOG:
            log.println(f"{optimized_lifetime:03}")

        return optimized_lifetime, optimize_equation, optimized_fail_transistor

    
    # for conf_index, conf_eq in enumerate(selector_conf):      # try all the SMs
    for conf_index, conf_eq in [(0, selector_conf[0])]:        # use the choosen SM
        alpha_cache = preload_alpha(conf_index)
        base_alpha = alpha_cache.get_cache("0")
        log.println(f"preload_alpha DONE + equations alphas")


        # optimize_sum_lifetime = multiprocessing.Value('d', 0.0)
        optimize_sum_lifetime = 0
        lock = multiprocessing.Lock()

        with multiprocessing.Pool(processes=PROCESS_POOL) as pool:
            results = pool.starmap(process_sample, [(i, base_alpha, conf_eq, bit_len) for i in range(SAMPLE)])

        for optimized_lifetime, optimize_equation, optimized_fail_transistor in results:
            # with lock:
                # optimize_sum_lifetime.value += optimized_lifetime
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



if True:

    log = Log(f"{__file__}.log", terminal=True)
    SAMPLE =  1000  # len(transistors) * samples

    def seed_generator(i):
        return 7*i + 1

    def process_sample(sample_index, base_alpha, SMs_alpha, bit_len):
        random_vth_matrix = generate_guassian_vth_base(seed=seed_generator(sample_index))
        

        base_lifetime = 0
        SMs_lifetime = 0
        special_SM_lifetime = 0
        
        vth_matrix_t0 = random_vth_matrix
        for event_time in [0, 50, 100, 150, 200, 1000]:
        # for event_time in [0, 100, 1000]:
            
            if base_lifetime >= event_time:
                base_lifetime = get_monte_carlo_life_expect(base_alpha, vth_matrix_t0, event_time, bit_len)["t_week"]
            
            if SMs_lifetime >= event_time:
                for sm_alpha in SMs_alpha:
                    SMs_lifetime = max(SMs_lifetime, get_monte_carlo_life_expect(sm_alpha, vth_matrix_t0, event_time, bit_len)["t_week"])
            
            if special_SM_lifetime >= event_time:
                special_SM_alpha = get_special_SM_alpha(vth_matrix_t0, event_time, False, bit_len)
                special_SM_lifetime = get_monte_carlo_life_expect(special_SM_alpha, vth_matrix_t0, event_time, bit_len)["t_week"]


            event_time_next = event_time + 50
            if (base_lifetime < event_time_next) and (SMs_lifetime < event_time_next) and (special_SM_lifetime < event_time_next):
                log.println(f"[{sample_index:6,}] {base_lifetime:3} -> {SMs_lifetime:3} /max: {special_SM_lifetime:3}")
                break

            """apply the event, update the Vth"""
            event_time_sec = event_time * 7 * 24 * 60 * 60
            event_time_next_sec = event_time_next * 7 * 24 * 60 * 60
            vth_matrix_t0 = BTI_aging_step(base_alpha, vth_matrix_t0, event_time_sec, event_time_next_sec)
            
            """a sudden 2% increase in three transistor"""
            random.seed(seed_generator(sample_index))
            for t in range(3):
                r_fa_i = random.randint(0, bit_len-2)
                r_fa_j = random.randint(0, bit_len-1)
                r_t_index = random.randint(0, 5)
                vth_matrix_t0[r_fa_i][r_fa_j][r_t_index] *= 1.02
                
                # log.println(f"EVENT")
                
    
    conf_index = 6 #1 OR 6
    conf_eq = selector_conf[conf_index]

    alpha_cache = preload_alpha(conf_index)
    base_alpha = alpha_cache.get_cache("0")
    # SMO_alpha = conf_eq[0]['alpha']
    SMs_alpha = []
    for conf in conf_eq:
        SMs_alpha += [conf['alpha']]
    log.println(f"preload_alpha DONE + equations alphas")

    
    
    for sample_index in range(SAMPLE):
        process_sample(sample_index, base_alpha, SMs_alpha, bit_len)



