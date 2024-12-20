import inspect
import random

from msimulator.get_alpha_class import MultiplierStressTest
from msimulator.Multiplier import *
from get_life_expect import get_life_expect

from tool.log import Log
log = Log(f"{__file__}.log", terminal=True)
log.println("START...")

bit_len = 8
# faulty_transistor = {'fa_i': 3, 'fa_j': 0, 't_index': 5, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
# faulty_transistor = {'fa_i': random.randint(0, bit_len-1-1), 'fa_j': random.randint(0, bit_len-1), 't_index': random.randint(0, 5), 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
# faulty_transistor = False
faulty_transistor = []

lst_transistor_optimize = []

# true if any of trans optimize is under stress
def optimizer_trigger(mp: MPn_v3, _a, _b):
    for transistor in lst_transistor_optimize:
        fa_i:int = transistor['fa_i']
        fa_j:int = transistor['fa_j']

        t_index:int = transistor['t_index']
        _tgate = t_index // 2
        _p = t_index % 2

        fa:FA = mp.gfa[fa_i][fa_j]

        if _p == 0:
            if fa.tgate[_tgate].p0.gate == L:
                return True
        else:
            if fa.tgate[_tgate].p1.gate == L:
                return True

    return False

# true if any of transistor is not under stree
def optimizer_accept(neg_mp: MPn_v3, _a, _b):
    for transistor in lst_transistor_optimize:
        fa_i:int = transistor['fa_i']
        fa_j:int = transistor['fa_j']

        t_index:int = transistor['t_index']
        _tgate = t_index // 2
        _p = t_index % 2

        fa:FA = neg_mp.gfa[fa_i][fa_j]
        
        ### OR
        if _p == 0:
            if fa.tgate[_tgate].p0.gate == H:
                return True
        else:
            if fa.tgate[_tgate].p1.gate == H:
                return True
    return False




# specific optimization
if False:
    lst_transistor_optimize = [{'fa_i': 1, 'fa_j': 7, 't_index': 1, 't_week': 98}, {'fa_i': 0, 'fa_j': 5, 't_index': 0, 't_week': 150}]
    # lst_transistor_optimize = []
    alpha_lst = MultiplierStressTest(bit_len, optimizer_trigger, optimizer_accept).run(log_obj=False)
    fail_transistor = get_life_expect(alpha_lst, bit_len, lst_transistor_optimize)
    # log.println("fault age of circuit")
    log.println(f"failed transistor: {fail_transistor}")
# DONE

if False:
    log.println(f"faulty transistor: {faulty_transistor}")
    for _ in range(5):
        alpha_lst = MultiplierStressTest(bit_len, optimizer_trigger, optimizer_accept).run()
        # log.println(f"alpha list: {alpha_lst}")

        fail_transistor = get_life_expect(alpha_lst, bit_len, faulty_transistor)
        log.println(f"failed transistor: {fail_transistor}")

        lst_transistor_optimize += [fail_transistor]
        log.println(f"optimization list: {lst_transistor_optimize}")



if False:
    # auto generating for all transistors
    max_life_time = 0
    min_life_time = 1000
    average_life_time = 0
    all_possible_faults = (bit_len - 1) * bit_len * 6

    DATASET_GENERATE = False
    MAX_OPT_LEVEL = 3

    for fa_i in range(bit_len - 1):
        for fa_j in range(bit_len):
            for t_index in range(6):
                
                faulty_transistor = {'fa_i': fa_i, 'fa_j': fa_j, 't_index': t_index, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
                log.println(f"faulty transistor: {faulty_transistor}")
                lst_transistor_optimize = []

                if DATASET_GENERATE:
                    data_set_log_name = f"dataset/fa_i-{fa_i}-fa_j-{fa_j}-t_index-{t_index}.txt"
                    data_set_log_obj = Log(data_set_log_name, terminal=False)
                
                    for i in range(MAX_OPT_LEVEL):
                        log.println(f"optimization list: {lst_transistor_optimize}")
                        if i != 2:
                            alpha_lst = MultiplierStressTest(bit_len, optimizer_trigger, optimizer_accept).run()
                        else:
                            alpha_lst = MultiplierStressTest(bit_len, optimizer_trigger, optimizer_accept).run(log_obj=data_set_log_obj)
                            del data_set_log_obj
                        fail_transistor = get_life_expect(alpha_lst, bit_len, faulty_transistor)
                        lst_transistor_optimize += [fail_transistor]
                
                else:
                    circuit_lifetime = 0
                    for i in range(MAX_OPT_LEVEL):
                        log.println(f"optimization list: {lst_transistor_optimize}")
                        alpha_lst = MultiplierStressTest(bit_len, optimizer_trigger, optimizer_accept).run()
                        fail_transistor = get_life_expect(alpha_lst, bit_len, faulty_transistor)
                        lst_transistor_optimize += [fail_transistor]

                        life_time = fail_transistor['t_week']
                        if life_time > circuit_lifetime:
                            circuit_lifetime = life_time
                    
                    average_life_time = circuit_lifetime / all_possible_faults
                    max_life_time = circuit_lifetime if circuit_lifetime > max_life_time else max_life_time
                    min_life_time = circuit_lifetime if circuit_lifetime < min_life_time else min_life_time

    log.println(f"max lifetime: {max_life_time}")
    log.println(f"min lifetime: {min_life_time}")
    log.println(f"average lifetime: {average_life_time}")



# ========================================================================================
# ========================================================================================
# ========================================================================================
if True:
    # compare equation lifetime with max optimization
    import re
    bit_len = 8

    def parse_pattern_line(line):
        pattern = r"\[\w+ \w+ \d+ \d+:\d+:\d+ \d+\] >> \[(.*?)\], \[(.*?)\], \[compliment: (True|False)\]"
        match = re.search(pattern, line)
        
        if match:
            A = list(map(int, match.group(1).split(", ")))
            B = list(map(int, match.group(2).split(", ")))
            # result = True if match.group(3) == "True" else False
            result = None
            if match.group(3) == 'True':
                result = True
            elif match.group(3) == 'False':
                result = False
            else:
                raise RuntimeError("invalid output in log file")
            
            return A, B, result
        return None

    def load_pattern_file(filepath):
        input_data = []
        
        with open(filepath, 'r') as file:
            for line in file:
                parsed_data = parse_pattern_line(line)
                if parsed_data:
                    input_data.append(parsed_data)
        
        return input_data

    input_data = []
    def dataset_optimizer_trigger(mp: MPn_v3, _a, _b):
        return True
    
    def dataset_optimizer_accept(neg_mp: MPn_v3, bin_A, bin_B):
        for data in input_data:
            A = data[0]
            B = data[1]

            if (A == bin_A) and (B == bin_B):
                return data[2]
        raise LookupError


    def eq_optimizer_trigger(mp: MPn_v3, A, B):
        return True

    def eq_optimizer_accept(neg_mp: MPn_v3, bin_A, bin_B):
        return False




    # maximum lifetime reading from dataset

    for fa_i in range(bit_len-1):
        for fa_j in range(bit_len):
            for t_index in range(6):

                log.println(f"faulty transistor: {(fa_i, fa_j, t_index)}")
                faulty_transistor = {'fa_i': fa_i, 'fa_j': fa_j, 't_index': t_index, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
                
                alpha_lst = MultiplierStressTest(bit_len, None, None, optimizer_enable=False).run(log_obj=False)
                fail_transistor = get_life_expect(alpha_lst, bit_len, faulty_transistor)
                lifetime = fail_transistor["t_week"]
                log.println(f"lifetime: {lifetime} weeks")

                data_set_log_name = f"dataset/fa_i-{fa_i}-fa_j-{fa_j}-t_index-{t_index}.txt"
                input_data=load_pattern_file(data_set_log_name)
                alpha_lst = MultiplierStressTest(bit_len, dataset_optimizer_trigger, dataset_optimizer_accept).run(log_obj=False)
                fail_transistor = get_life_expect(alpha_lst, bit_len, faulty_transistor)
                lifetime = fail_transistor["t_week"]
                log.println(f"dataset lifetime: {lifetime} weeks (maximum optimization) \n")