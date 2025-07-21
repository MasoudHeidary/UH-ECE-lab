


from tool.log import Log

from msimulator.get_alpha_class import MultiplierStressTest
from msimulator.Multiplier import *

from msimulator.Multiplier import MPn_v3
from get_life_expect import *

import tool.NBTI_formula as NBTI
import tool.vth_body_map as VTH
from tool.map_pb_to_current import get_current_from_pb, get_pb_from_current

# cache
import pickle
from random import randint

bit_len = 8
log = Log(f"{__file__}.log", terminal=True)


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

try:
    alpha_cache = CACHE.load_cache(filename=f"{__file__}.cache")
except:
    alpha_cache = CACHE(filename=f"{__file__}.cache")

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
### monte carlo sampling computations
##############################

def get_monte_carlo_life_expect(alpha, vth_matrix_t0, t0_week, bit_len):
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


def get_F2F_lst(vth_matrix, log=False, bit_len=bit_len):
    global lst_transistor_optimize
    lst_transistor_optimize = []

    for i in range(10):
        if len(lst_transistor_optimize) != i:
            break
        
        if alpha_cache.hit_cache(str(lst_transistor_optimize)):
            alpha = alpha_cache.get_cache(str(lst_transistor_optimize))
        else:
            alpha = MultiplierStressTest(bit_len, optimizer_trigger, optimizer_accept, True).run()
            alpha_cache.add_cache(str(lst_transistor_optimize), alpha)

        fail_transistor = get_monte_carlo_life_expect(alpha, vth_matrix, 0, bit_len)
        fail_transistor.pop("t_week")

        if fail_transistor in lst_transistor_optimize:
            break

        lst_transistor_optimize.append(fail_transistor)
        lst_transistor_optimize = list(lst_transistor_optimize)

    return lst_transistor_optimize

def compare_F2F(new_f2f, olf_f2f):
    """true even if new f2f is a subset"""
    for v in new_f2f:
        if v not in olf_f2f:
            return False
    return True

# multi process monte carlo real simulation
if True:

    PROCESS_POOL = 1
    log = Log(f"{__file__}.log", terminal=True)
    # SAMPLE =  336 * 1000  # len(transistors) * samples
    SAMPLE =  5000  # len(transistors) * samples
    DETAIL_LOG = False
    CHART = True

    no_mitigation_alpha = MultiplierStressTest(bit_len, optimizer_trigger, optimizer_accept, True).run()

    def seed_generator(i):
        return 7*i + 1
    

    def process_sample(sample_index, bit_len):
        """Function to process a single sample in parallel"""
        random_vth_matrix = generate_guassian_vth_base(seed=seed_generator(sample_index))
        

        max_optimizer_lifetime = 0
        max_failed_transistor = None

        """to check if the event caused the circuit to change the failed transistor location"""
        original_F2F = None
        
        vth_matrix_t0 = random_vth_matrix
        for event_time in [0, 100]:
            event_time_next = event_time + 100

            if event_time == 0:
                original_F2F = get_F2F_lst(vth_matrix_t0, log, bit_len)
                log.println(f"[{sample_index}] original F2F: {original_F2F}")

            else:
                """restart the max lifetime numbers"""
                max_optimizer_lifetime = 0
                max_failed_transistor = None
                
                """apply the event, update the Vth"""
                event_time_sec = event_time * 7 * 24 * 60 * 60
                event_time_next_sec = event_time_next * 7 * 24 * 60 * 60
                vth_matrix_t0 = BTI_aging_step(no_mitigation_alpha, vth_matrix_t0, event_time_sec, event_time_next_sec)
                
                """a sudden 2% increase in three transistor"""
                random.seed(seed_generator(sample_index))
                for t in range(3):
                    r_fa_i = random.randint(0, bit_len-2)
                    r_fa_j = random.randint(0, bit_len-1)
                    r_t_index = random.randint(0, 5)
                    vth_matrix_t0[r_fa_i][r_fa_j][r_t_index] *= 1.02

                
                """check to see if F2F has changed after the event"""
                new_F2F = get_F2F_lst(vth_matrix_t0, log, bit_len)
                if not compare_F2F(new_F2F, original_F2F):
                    log.println(f"event {event_time} {original_F2F} -> {new_F2F} [{False}]")
                    break
                
            


    for i in range(SAMPLE):
        process_sample(i, bit_len)

    
