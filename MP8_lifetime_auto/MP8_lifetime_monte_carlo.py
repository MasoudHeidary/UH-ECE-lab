


from tool.log import Log

from msimulator.get_alpha_class import MultiplierStressTest
from msimulator.Multiplier import MPn_v3
from get_life_expect import get_life_expect
from sympy import symbols, Or, And, Not, simplify_logic
from pyeda.inter import expr

# cache
import pickle
from random import randint


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

### no optimization
# equation_conf = [
#     {
#         'equation': '0',
#         'transistor_list': [
#             (fa_i, fa_j, t_index) for fa_i in range(bit_len-1) for fa_j in range(bit_len) for t_index in range(6)
#         ],
#         'alpha': None
#     }
# ]

### healthy optimizer
equation_conf = [
    {
        'equation': '(B0 & ~A6) | (B0 & ~A7 & ~B1) | (A6 & B1 & ~A7 & ~B0)',
        'transistor_list': [
            (fa_i, fa_j, t_index) for fa_i in range(bit_len-1) for fa_j in range(bit_len) for t_index in range(6)
        ],
        'alpha': None
    }
]



##############################
### ideal computations
##############################

if True:
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
        log.println(f"conf/{conf_index} transistor len: {len_transistor}")

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
    log.println(f"conf:\n{equation_conf}\n")
    log.println(f"final result >>> sum lifetime {sum_lifetime} / transistor len {len_transistor} = {sum_lifetime/len_transistor}")







##############################
### monte carlo sampling computations
##############################