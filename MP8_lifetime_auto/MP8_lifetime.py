import inspect
import random

from msimulator.get_alpha_class import MultiplierStressTest
from msimulator.Multiplier import *
from get_life_expect import get_life_expect

from tool.log import Log
log = Log("MP8_lifetime.txt", terminal=True)
log.println()

bit_len = 8
# faulty_transistor = {'fa_i': 3, 'fa_j': 0, 't_index': 5, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
# faulty_transistor = {'fa_i': random.randint(0, bit_len-1-1), 'fa_j': random.randint(0, bit_len-1), 't_index': random.randint(0, 5), 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
faulty_transistor = False
log.println(f"faulty transistor: {faulty_transistor}")

lst_transistor_optimize = []

def optimizer_trigger(mp: MPn_v3, A:int, B:int):
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

def optimizer_accept(neg_mp: MPn_v3, neg_A:int, neg_B:int):
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


"""
def optimizer_accept(neg_mp: MPn_v3, neg_A:int, neg_B:int):
    for transistor in lst_transistor_optimize:
        fa_i:int = transistor['fa_i']
        fa_j:int = transistor['fa_j']

        t_index:int = transistor['t_index']
        _tgate = t_index // 2
        _p = t_index % 2

        fa:FA = neg_mp.gfa[fa_i][fa_j]
            
        ### AND
        if _p == 0:
            if fa.tgate[_tgate].p0.gate != H:
                return False
        else:
            if fa.tgate[_tgate].p1.gate != H:
                return False
        
    return True  
"""  


# log.println("optimization trigger")
# log.println(inspect.getsource(optimizer_trigger))
# log.println("optimization accept")
# log.println(inspect.getsource(optimizer_accept))

for _ in range(5):
    alpha_lst = MultiplierStressTest(bit_len, optimizer_trigger, optimizer_accept).run()
    # log.println(f"alpha list: {alpha_lst}")

    fail_transistor = get_life_expect(alpha_lst, bit_len, faulty_transistor)
    log.println(f"failed transistor: {fail_transistor}")

    lst_transistor_optimize += [fail_transistor]
    # log.println(f"optimization list: {lst_transistor_optimize}")
