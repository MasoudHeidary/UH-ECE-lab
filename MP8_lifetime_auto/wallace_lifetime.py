



from msimulator.Multiplier import *
from msimulator.bin_func import *
from tool.log import Log, Progress
from msimulator.get_alpha_wallace_shrinked import AlphaWallaceShrinked

from get_life_expect import get_life_expect

log = Log(f"{__file__}.log", terminal=True)
bar = Progress(bars=1)

def wallace_alpha(raw_mp, bit_len, op_trigger=None, op_accept=None, op_enable=False, log_obj: Log=False):
    alpha_row = bit_len-1
    alpha_index = bit_len
    alpha = [[[0 for _ in range(6)] for _ in range(alpha_index)] for _ in range(alpha_row)]

    limit = 2 ** (bit_len-1)
    for a in range(-limit, limit):
        bar.keep_line()
        bar.update(0, (a+limit)/(2*limit-1))
        for b in range(-limit, limit):
            
            a_bin = signed_b(a, bit_len)
            b_bin = signed_b(b, bit_len)
            
            mp: Wallace_comp
            mp = raw_mp(a_bin, b_bin, bit_len)
            mp.output

            ### selector pattern
            select_flag = False
            if op_enable:
                if (a != -limit) and (b != -limit):
                    if (op_trigger(mp)):
                        neg_a = -a
                        neg_b = -b
                        neg_mp = raw_mp(
                            signed_b(neg_a, bit_len),
                            signed_b(neg_b, bit_len),
                            bit_len
                        )
                        neg_mp.output

                        if op_accept(neg_mp):
                            select_flag = True
                            mp = neg_mp


            if log_obj:
                log_obj.println(f"{a_bin}, {b_bin}, [compliment: {select_flag}]")
            
            ### END 

            # update alpha as counter

            for row in range(alpha_row):
                for index in range(alpha_index):
                    for t in range(6):
                        alpha[row][index][t] += (not mp.gfa[row][index].p[t])
    
    # alpha counter -> alpha probability
    for row in range(alpha_row):
        for index in range(alpha_index):
            for t in range(6):
                alpha[row][index][t] /= ((2 * limit) ** 2)

    return alpha

critical_transistor_lst = []
def op_trigger(mp: Wallace_comp):
    for critical_t in critical_transistor_lst:
        fa_i = critical_t['fa_i']
        fa_j = critical_t['fa_j']
        t_index = critical_t['t_index']

        if mp.gfa[fa_i][fa_j].p[t_index] == L:
            return True
        
    return False

def op_accept(neg_mp: Wallace_comp):
    for critical_t in critical_transistor_lst:
        fa_i = critical_t['fa_i']
        fa_j = critical_t['fa_j']
        t_index = critical_t['t_index']

        if neg_mp.gfa[fa_i][fa_j].p[t_index] == H:
            return True
        
    return False


if True:
    """same experiment but with shrinked alpha for faster computations [no lookup table]"""

    BIT_LEN = 16
    # SAMPLE_COUNT = 100_000_000
    SAMPLE_COUNT = 1_000_000
    RND_SEED = 7

    faulty_transistor = []
    critical_transistor_lst = []
    
    log.println(f"faulty transistor: {faulty_transistor}")
    for _ in range(3):
        # alpha_lst = AlphaShrinked(bit_len, optimizer_trigger, optimizer_accept).run(sample_count=SAMPLE_COUNT, rnd_seed=RND_SEED, log_obj=log)
        alpha_lst = AlphaWallaceShrinked(BIT_LEN, op_trigger, op_accept).run_multi(sample_count=SAMPLE_COUNT, rnd_seed=RND_SEED, log_obj=log)
        log.println(f"alpha list: {alpha_lst}")

        fail_transistor = get_life_expect(alpha_lst, BIT_LEN, faulty_transistor)
        log.println(f"[{BIT_LEN}] failed transistor: {fail_transistor}")

        critical_transistor_lst += [fail_transistor]
        log.println(f"optimization list: {critical_transistor_lst}")



if False:
    """
        NOTE:

        8-bit wallace multiplier
        critical transistors in order
        failed transistor: {'fa_i': 6, 'fa_j': 7, 't_index': 3, 't_week': 83}
        failed transistor: {'fa_i': 0, 'fa_j': 6, 't_index': 0, 't_week': 132}
        failed transistor: {'fa_i': 6, 'fa_j': 7, 't_index': 3, 't_week': 132}  
        // DONE

        4-bit wallace multiplier
        {'fa_i': 2, 'fa_j': 3, 't_index': 3, 't_week': 105}
        {'fa_i': 0, 'fa_j': 2, 't_index': 0, 't_week': 141}
        {'fa_i': 2, 'fa_j': 3, 't_index': 3, 't_week': 145}
        {'fa_i': 2, 'fa_j': 3, 't_index': 3, 't_week': 145}
        // DONE

        12-bit wallace multiplier
        {'fa_i': 10, 'fa_j': 11, 't_index': 3, 't_week': 82}
        {'fa_i': 0, 'fa_j': 10, 't_index': 0, 't_week': 131}
        {'fa_i': 10, 'fa_j': 11, 't_index': 3, 't_week': 131}
        // DONE
        
    """

    BIT_LEN = 8
    log.println(f"RUN >> [{BIT_LEN}] bit")

    # critical_transistor_lst = []
    critical_transistor_lst = [
        {'fa_i': 6, 'fa_j': 7, 't_index': 3, 't_week': 83},
        {'fa_i': 0, 'fa_j': 6, 't_index': 0, 't_week': 132},
        {'fa_i': 6, 'fa_j': 7, 't_index': 3, 't_week': 132},
    ]

    # pattern_file = Log("pattern-wallace8.txt")
    
    alpha = wallace_alpha(Wallace_comp, BIT_LEN, op_trigger, op_accept, op_enable=True, log_obj=False)
    fail_transistor = get_life_expect(alpha, BIT_LEN, faulty_transistor=False)
    
    log.println(f"RES: BIT_LEN, critical_transistor_lst")
    log.println(f"RES: [{BIT_LEN}] >> {critical_transistor_lst}")
    log.println(f"failed transistor: {fail_transistor}")



    if True:
        unoptimized_alpha = wallace_alpha(Wallace_comp, BIT_LEN, None, None, op_enable=False)
        for fa_i in range(BIT_LEN - 1):
            for fa_j in range(BIT_LEN):
                for t_index in range(6):
                    faulty_transistor = {'fa_i': fa_i, 'fa_j': fa_j, 't_index': t_index, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
                    unoptimized_lifetime = get_life_expect(unoptimized_alpha, BIT_LEN, faulty_transistor)["t_week"]
                    lookup_table_lifetime = get_life_expect(alpha, BIT_LEN, faulty_transistor)["t_week"]

                    log.println(f"faulty transistor: {faulty_transistor} >>> using normal lookup table >>> {unoptimized_lifetime:03} -> {lookup_table_lifetime:03}")



