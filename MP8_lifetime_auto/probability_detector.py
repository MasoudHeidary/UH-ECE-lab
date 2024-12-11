"""
NOTE:


for normal aging the pattern is:

CODE:
r = calc_accuracy(
    input_data,
    ['x', 'x', 'x', 'x', 'x', 'x', 'b', 'b'],
    ['b', 'b', 'x', 'x', 'x', 'x', 'x', 'x'],
    log_obj = log
)
log.println(f"{r}")

OUTPUT:
TRAINING...
['x', 'x', 'x', 'x', 'x', 'x', 0, 0] [0, 0, 'x', 'x', 'x', 'x', 'x', 'x']: (0, 4096) 0.0 [False]
['x', 'x', 'x', 'x', 'x', 'x', 0, 0] [0, 1, 'x', 'x', 'x', 'x', 'x', 'x']: (0, 4096) 0.0 [False]
['x', 'x', 'x', 'x', 'x', 'x', 0, 0] [1, 0, 'x', 'x', 'x', 'x', 'x', 'x']: (4032, 64) 63.0 [True]
['x', 'x', 'x', 'x', 'x', 'x', 0, 0] [1, 1, 'x', 'x', 'x', 'x', 'x', 'x']: (4032, 64) 63.0 [True]
['x', 'x', 'x', 'x', 'x', 'x', 0, 1] [0, 0, 'x', 'x', 'x', 'x', 'x', 'x']: (0, 4096) 0.0 [False]
['x', 'x', 'x', 'x', 'x', 'x', 0, 1] [0, 1, 'x', 'x', 'x', 'x', 'x', 'x']: (0, 4096) 0.0 [False]
['x', 'x', 'x', 'x', 'x', 'x', 0, 1] [1, 0, 'x', 'x', 'x', 'x', 'x', 'x']: (4032, 64) 63.0 [True]
['x', 'x', 'x', 'x', 'x', 'x', 0, 1] [1, 1, 'x', 'x', 'x', 'x', 'x', 'x']: (4032, 64) 63.0 [True]
['x', 'x', 'x', 'x', 'x', 'x', 1, 0] [0, 0, 'x', 'x', 'x', 'x', 'x', 'x']: (0, 4096) 0.0 [False]
['x', 'x', 'x', 'x', 'x', 'x', 1, 0] [0, 1, 'x', 'x', 'x', 'x', 'x', 'x']: (4032, 64) 63.0 [True]
['x', 'x', 'x', 'x', 'x', 'x', 1, 0] [1, 0, 'x', 'x', 'x', 'x', 'x', 'x']: (4096, 0) 4096000.0 [True]
['x', 'x', 'x', 'x', 'x', 'x', 1, 0] [1, 1, 'x', 'x', 'x', 'x', 'x', 'x']: (64, 4032) 0.0 [False]
['x', 'x', 'x', 'x', 'x', 'x', 1, 1] [0, 0, 'x', 'x', 'x', 'x', 'x', 'x']: (0, 4096) 0.0 [False]
['x', 'x', 'x', 'x', 'x', 'x', 1, 1] [0, 1, 'x', 'x', 'x', 'x', 'x', 'x']: (0, 4096) 0.0 [False]
['x', 'x', 'x', 'x', 'x', 'x', 1, 1] [1, 0, 'x', 'x', 'x', 'x', 'x', 'x']: (64, 4032) 0.0 [False]
['x', 'x', 'x', 'x', 'x', 'x', 1, 1] [1, 1, 'x', 'x', 'x', 'x', 'x', 'x']: (0, 4096) 0.0 [False]
(0.9931640625, 24256, 40832, 320, 128)

['x', 'x', 'x', 'x', 'x', 'x', 'b', 'b']  ['b', 'b', 'x', 'x', 'x', 'x', 'x', 'x']
Accuracy:  0.993, TP: 24256, TN: 40832, FP(not suppose compliment): 320, FN(lost compliments):128,




"""


from itertools import *
import multiprocessing
import re
import itertools
import time
from tool.log import Log

log = Log(f"{__file__}.log")

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

def signed_b(num: int, bit_len: int):
    num_cpy = num
    if num < 0:
        num_cpy = 2**bit_len + num
    bit_num = list(map(int, reversed(format(num_cpy, f'0{bit_len}b'))))

    if (num > 0) and (bit_num[-1] != 0):
        raise OverflowError(f"number {num} can't fit in signed #{bit_len} bits")
    if (num < 0) and (bit_num[-1] != 1):
        raise OverflowError(f"number {num} can't fit in signed #{bit_len} bits")
    return bit_num

def reverse_signed_b(binary_list):
    binary_str = ''.join(map(str, reversed(binary_list)))
    num = int(binary_str, 2)

    if binary_list[-1] == 1:
        num = num - (2**len(binary_list))
    return num





def comp(x_data, solid_data):
    if len(x_data) != len(solid_data):
        raise RuntimeError()
    
    for i, value in enumerate(x_data):
        if value == 'x':
            continue
        if value != solid_data[i]:
            return False
    
    return True


def calc_probability(input_data, partial_A, partial_B):
    # global input_data
    false_count = 0
    true_count = 0

    for data in input_data:
        A = data[0]
        B = data[1]
        out = data[2]

        if (partial_A == A[:len(partial_A)]) and (partial_B == B[:len(partial_B)]) and (out == True):
            true_count += 1
        elif (partial_A == A[:len(partial_A)]) and (partial_B == B[:len(partial_B)]) and (out == False):
            false_count += 1

    return true_count, false_count     

def calc_prob(input_data, partial_A, partial_B):
    # global input_data
    false_count = 0
    true_count = 0

    for data in input_data:
        A = data[0]
        B = data[1]
        out = data[2]

        if comp(partial_A, A) and comp(partial_B, B) and (out == True):
            true_count += 1
        elif comp(partial_A, A) and comp(partial_B, B) and (out == False):
            false_count += 1

    return true_count, false_count 

def generate_patterns(bit_pattern):
    b_positions = [i for i, value in enumerate(bit_pattern) if value == 'b']
    combinations = itertools.product([0,1], repeat=len(b_positions))
    result = []
    for comb in combinations:
        temp_list = bit_pattern[:]
        for idx, value in zip(b_positions, comb):
            temp_list[idx] = value
        result.append(temp_list)
    return result


def interface(table, data):
    for tb in table:
        t_A = tb['A']
        t_B = tb['B']

        if comp(t_A, data[0]) and comp(t_B, data[1]):
            t_out = tb['out']
            return t_out
    raise RuntimeError("table FAILED!")


def calc_accuracy(input_data, A_bit_pattern, B_bit_pattern, log_obj=False):
    # global input_data

    # traing
    if log_obj:
        log_obj.println("TRAINING...")
    A_patterns = generate_patterns(A_bit_pattern)
    B_patterns = generate_patterns(B_bit_pattern)

    ML_table = []

    for a in A_patterns:
        for b in B_patterns:
            true_prob, false_prob = calc_prob(input_data, a, b)
            if log_obj:
                log_obj.println(f"{a} {b}: {true_prob, false_prob} {true_prob/(false_prob or 0.001):.1f} [{'True' if true_prob > false_prob else 'False'}]")

            ML_table.append(
                {'A': a, 'B': b, 'out': true_prob > false_prob}
            )

    # testing
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for data in input_data:
        y_pred = interface(ML_table, data)
        d_out = data[2]

        if (y_pred == d_out) and (y_pred == True):
            true_positive += 1
        elif (y_pred == d_out) and (y_pred == False):
            true_negative += 1
        elif (y_pred != d_out) and (y_pred == True):
            false_positive += 1
        elif (y_pred != d_out) and (y_pred == False):
            false_negative += 1

    # accu = true_positive / ((true_positive + false_positive) or 1)
    accu = (true_positive + true_negative) / ((true_positive + true_negative + false_positive + false_negative) or 1)
    # if log_obj:
    #     log_obj.println(f"{A_bit_pattern}\t{B_bit_pattern} \t\t Accuracy: {accu: 0.3f}, TP: {true_positive}, TN: {true_negative}, FP(not suppose compliment): {false_positive}, FN(lost compliments):{false_negative},")
        # log.println(f"Accuracy: {accu: 0.3f}, TP: {true_positive}, TN: {true_negative}, FP(not suppose compliment): {false_positive}, FN(lost compliments):{false_negative}, ")
    return accu, true_positive, true_negative, false_positive, false_negative




def multi_process_best_pattern_finder(
        input_data,
        lst_count_A_bit_pattern,
        lst_count_B_bit_pattern,
        max_process_count = multiprocessing.cpu_count(),
        log_obj = False,
):
    def work_wrapper(input_data, a_pattern, b_pattern, log_obj, results):
        accu, TP, TN, FP, FN = calc_accuracy(input_data, a_pattern, b_pattern, log_obj=False)
        if log_obj:
            log_obj.println(f"{a_pattern}\t{b_pattern}\t{accu} ({(TP, TN, FP, FN)})")
        results.append({
            'A': a_pattern,
            'B': b_pattern,
            'accu': accu,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        })

    processes = []
    processes_count = 0
    result = multiprocessing.Manager().list()

    A_pattern = list(product(['b', 'x'], repeat=8))
    filtered_A = [lst for lst in A_pattern if lst.count('b') in lst_count_A_bit_pattern]
    filtered_A = [list(lst) for lst in filtered_A]

    for a_pattern in filtered_A:
        B_pattern = list(product(['b', 'x'], repeat=8))
        filtered_B = [lst for lst in B_pattern if lst.count('b') in lst_count_B_bit_pattern]
        filtered_B = [list(lst) for lst in filtered_B]

        for b_pattern in filtered_B:
            processes.append(
                multiprocessing.Process(
                    target=work_wrapper,
                    args=(input_data, a_pattern, b_pattern, log_obj, result),
                )
            )
            processes_count += 1

            if len(processes) == max_process_count:
                for p in processes:
                    p.start()
                for p in processes:
                    p.join()

                processes = []
    
    # process left over processes
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    processes = []

    # best pattern
    max_accuracy = max([i['accu'] for i in result])
    max_accuracy_patterns = [i for i in result if i['accu']==max_accuracy]
    best_accuracy_pattern = max_accuracy_patterns[0]
    for pattern in max_accuracy_patterns:
        if  (
            pattern['A'].count('b') + pattern['B'].count('b')
            ) < (
            best_accuracy_pattern['A'].count('b') + best_accuracy_pattern['B'].count('b')
            ):
            best_accuracy_pattern = pattern
    
    if log_obj:
        log_obj.println(f"max accuracy: {max_accuracy}")
        log_obj.println(f"processes count: {processes_count}")
    return best_accuracy_pattern








# =========================================================================================================
# =========================================================================================================
# =========================================================================================================

log_filepath = 'pattern.txt'
input_data = load_pattern_file(log_filepath)

count = 0
for data in input_data:
    count += 1 if data[2] else 0
log.println(f"number of TRUE patterns: {count}")


_start_time = time.time()

r = multi_process_best_pattern_finder(
        input_data,
        [2],
        [2],
        log_obj=False,
    )
log.println(f"{r}")

_end_time = time.time()
log.println(f"execution time: {_end_time - _start_time}")
