import inspect
import random

from msimulator.get_alpha_PE import SemiPEArrayStressTest
from msimulator.semi_PE import *
from get_life_PE import get_life_pe

from tool.log import Log
log = Log("PE_lifetime.txt", terminal=True)
log.println()

x_len = 1
y_len = 2
bit_len = 8
# faulty_transistor = {'fa_i': 3, 'fa_j': 0, 't_index': 5, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
# faulty_transistor = {'fa_i': random.randint(0, bit_len-1-1), 'fa_j': random.randint(0, bit_len-1), 't_index': random.randint(0, 5), 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
faulty_transistor = False

lst_transistor_optimize = []


def get_lst_dimensions(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_lst_dimensions(lst[0]) if lst else []
    return []

def optimizer_trigger(pe: SemiPEArray):
    for transistor in lst_transistor_optimize:
        pe_x:int = transistor['pe_x']
        pe_y:int = transistor['pe_y']
        fa_i:int = transistor['fa_i']
        fa_j:int = transistor['fa_j']
        t_index:int = transistor['t_index']
        _tgate = t_index // 2
        _p = t_index % 2

        fa:FA = pe.mp[pe_x][pe_y].gfa[fa_i][fa_j]

        if _p == 0:
            if fa.tgate[_tgate].p0.gate == L:
                return True
        else:
            if fa.tgate[_tgate].p1.gate == L:
                return True

    return False

def optimizer_accept(neg_pe: SemiPEArray):
    for transistor in lst_transistor_optimize:
        pe_x:int = transistor['pe_x']
        pe_y:int = transistor['pe_y']
        fa_i:int = transistor['fa_i']
        fa_j:int = transistor['fa_j']
        t_index:int = transistor['t_index']
        _tgate = t_index // 2
        _p = t_index % 2

        fa:FA = neg_pe.mp[pe_x][pe_y].gfa[fa_i][fa_j]
        
        ### OR
        if _p == 0:
            if fa.tgate[_tgate].p0.gate == H:
                return True
        else:
            if fa.tgate[_tgate].p1.gate == H:
                return True
    return False


# log.println("optimization trigger")
# log.println(inspect.getsource(optimizer_trigger))
# log.println("optimization accept")
# log.println(inspect.getsource(optimizer_accept))


if True:
    log.println(f"faulty transistor: {faulty_transistor}")
    for _ in range(5):
        alpha_lst = SemiPEArrayStressTest(x_len, y_len, bit_len, optimizer_trigger, optimizer_accept).run()
        # log.println(f"alpha list: {alpha_lst}")
        log.println(f"alpha_lst dimention {get_lst_dimensions(alpha_lst)}")

        fail_transistor = get_life_pe(alpha_lst, x_len, y_len, bit_len, faulty_transistor)
        log.println(f"failed transistor: {fail_transistor}")

        lst_transistor_optimize += [fail_transistor]
        log.println(f"optimization list: {lst_transistor_optimize}")


# auto generating for all transistors
if False:
    for pe_x in range(x_len):
        for pe_y in range(y_len):

            for fa_i in range(bit_len - 1):
                for fa_j in range(bit_len):
                    
                    for t_index in range(6):
                        
                        faulty_transistor = {'fa_i': fa_i, 'fa_j': fa_j, 't_index': t_index, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
                        log.println(f"faulty transistor: {faulty_transistor}")
                        lst_transistor_optimize = []

                        data_set_log_name = f"dataset/fa_i-{fa_i}-fa_j-{fa_j}-t_index-{t_index}.txt"
                        data_set_log_obj = Log(data_set_log_name, terminal=False)
                        
                        for i in range(3):
                            log.println(f"optimization list: {lst_transistor_optimize}")
                            if i != 2:
                                alpha_lst = MultiplierStressTest(bit_len, optimizer_trigger, optimizer_accept).run()
                            else:
                                alpha_lst = MultiplierStressTest(bit_len, optimizer_trigger, optimizer_accept).run(log_obj=data_set_log_obj)
                                del data_set_log_obj
                            fail_transistor = get_life_expect(alpha_lst, bit_len, faulty_transistor)
                            lst_transistor_optimize += [fail_transistor]



