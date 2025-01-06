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
import math
from tool.log import Log
from sympy import symbols, Or, And, Not, simplify_logic
from pyeda.inter import expr
from collections import Counter
import ast
from msimulator.get_alpha_class import MultiplierStressTest
from msimulator.Multiplier import MPn_v3
from get_life_expect import get_life_expect

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


def truth_table(input_data, A_bit_pattern, B_bit_pattern, log_obj=False):
    if log_obj:
        log.println("Truth Table >> Training")
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

    return ML_table

def generate_equation(truth_table):
    terms = []
    variables = [
        'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
        'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
        ]
    for row in truth_table:
        inputs, output = (row['A']+row['B']), row['out']
        if output == 1:
            term = []
            for var, val in zip(variables, inputs):
                if val != 'x':  # Ignore 'x' bits
                    term.append(f"{var}" if val == 1 else f"~{var}")
            terms.append(" & ".join(term))

    equation = " | ".join(f"({term})" for term in terms)
    return equation if equation else "0"



def generate_optimized_equation_with_or(truth_table):
    variables = symbols('A0 A1 A2 A3 A4 A5 A6 A7 B0 B1 B2 B3 B4 B5 B6 B7')
    minterms = []
    for row in truth_table:
        inputs, output = (row['A'] + row['B']), row['out']
        if output == 1:
            term = []
            for var, val in zip(variables, inputs):
                if val != 'x':
                    term.append(var if val == 1 else Not(var))
            minterms.append(And(*term))

    equation = Or(*minterms)
    optimized_equation = simplify_logic(equation, form='dnf')

    return str(optimized_equation) if optimized_equation else "0"

def generate_optimized_equation_with_and(truth_table):
    variables = symbols('A0 A1 A2 A3 A4 A5 A6 A7 B0 B1 B2 B3 B4 B5 B6 B7')
    maxterms = []
    for row in truth_table:
        inputs, output = (row['A'] + row['B']), row['out']
        if output == 0:  # Collect maxterms where output is 0
            term = []
            for var, val in zip(variables, inputs):
                if val != 'x':
                    # If the value is 1, use Not(var); if 0, use var itself
                    term.append(var if val == 0 else Not(var))
            maxterms.append(Or(*term))

    # Combine all maxterms using AND
    equation = And(*maxterms)
    optimized_equation = simplify_logic(equation, form='cnf')

    return str(optimized_equation) if optimized_equation else "1"


def calc_accuracy(input_data, A_bit_pattern, B_bit_pattern, log_obj=False):
    ML_table = truth_table(input_data, A_bit_pattern, B_bit_pattern, log_obj)

    if log_obj:
        log_obj.println(f"Calc Accuracy >> testing")

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

    accu = (true_positive + true_negative) / ((true_positive + true_negative + false_positive + false_negative) or 1)
    if log_obj:
        log_obj.println(f"{A_bit_pattern}\t{B_bit_pattern}")
        log_obj.println(f"Accuracy: {accu: 0.3f}, TP: {true_positive}, TN: {true_negative}, FP(not suppose compliment): {false_positive}, FN(lost compliments):{false_negative},")
    return accu, true_positive, true_negative, false_positive, false_negative




# def multi_process_best_pattern_finder(
#         input_data,
#         lst_count_A_bit_pattern,
#         lst_count_B_bit_pattern,
#         max_process_count = multiprocessing.cpu_count(),
#         log_obj = False,
# ):
#     if log_obj:
#         _start_time = time.time()

#     def work_wrapper(input_data, a_pattern, b_pattern, log_obj, results):
#         accu, TP, TN, FP, FN = calc_accuracy(input_data, a_pattern, b_pattern, log_obj=False)
#         if log_obj:
#             log_obj.println(f"{a_pattern}\t{b_pattern}\t{accu} ({(TP, TN, FP, FN)})")
#         results.append({
#             'A': a_pattern,
#             'B': b_pattern,
#             'accu': accu,
#             'TP': TP,
#             'TN': TN,
#             'FP': FP,
#             'FN': FN
#         })

#     processes = []
#     processes_count = 0
#     result = multiprocessing.Manager().list()

#     A_pattern = list(product(['b', 'x'], repeat=8))
#     filtered_A = [lst for lst in A_pattern if lst.count('b') in lst_count_A_bit_pattern]
#     filtered_A = [list(lst) for lst in filtered_A]

#     for a_pattern in filtered_A:
#         B_pattern = list(product(['b', 'x'], repeat=8))
#         filtered_B = [lst for lst in B_pattern if lst.count('b') in lst_count_B_bit_pattern]
#         filtered_B = [list(lst) for lst in filtered_B]

#         for b_pattern in filtered_B:
#             processes.append(
#                 multiprocessing.Process(
#                     target=work_wrapper,
#                     args=(input_data, a_pattern, b_pattern, False, result),
#                 )
#             )
#             processes_count += 1

#             if len(processes) == max_process_count:
#                 for p in processes:
#                     p.start()
#                 for p in processes:
#                     p.join()

#                 processes = []
    
#     # process left over processes
#     for p in processes:
#         p.start()
#     for p in processes:
#         p.join()
#     processes = []

#     # best pattern
#     max_accuracy = max([i['accu'] for i in result])
#     max_accuracy_patterns = [i for i in result if i['accu']==max_accuracy]
#     best_accuracy_pattern = max_accuracy_patterns[0]
#     for pattern in max_accuracy_patterns:
#         if  (
#             pattern['A'].count('b') + pattern['B'].count('b')
#             ) < (
#             best_accuracy_pattern['A'].count('b') + best_accuracy_pattern['B'].count('b')
#             ):
#             best_accuracy_pattern = pattern
    
#     if log_obj:
#         log_obj.println(f"max accuracy: {max_accuracy}")
#         log_obj.println(f"processes count: {processes_count}")

#         _end_time = time.time()
#         _func_time = _end_time - _start_time
#         log.println(f"execution time:\t{_func_time}s")
#     return best_accuracy_pattern


def multi_process_best_pattern_finder(
        input_data,
        count_bit_pattern,
        max_process_count = multiprocessing.cpu_count(),
        log_obj = False,
):
    if log_obj:
        _start_time = time.time()

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
    A_pattern = [list(lst) for lst in A_pattern]

    for a_pattern in A_pattern:
        B_pattern = list(product(['b', 'x'], repeat=8))
        B_pattern = [list(lst) for lst in B_pattern]

        for b_pattern in B_pattern:
            if a_pattern.count('b') + b_pattern.count('b') != count_bit_pattern:
                continue

            processes.append(
                multiprocessing.Process(
                    target=work_wrapper,
                    args=(input_data, a_pattern, b_pattern, False, result),
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
        log_obj.println(f"**max accuracy: {max_accuracy}")
        log_obj.println(f"**processes count: {processes_count}")

        _end_time = time.time()
        _func_time = _end_time - _start_time
        log.println(f"**execution time:\t{_func_time}s")
    return best_accuracy_pattern




# =========================================================================================================
# =========================================================================================================
# =========================================================================================================

if False:
    # normal agine, fine logical optimizer
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

    exit()

# input_data = load_pattern_file(f"pattern.txt")
# log = Log(f"{__file__}.{0}.{0}.{1}.txt")

# TT = truth_table(
#         input_data,
#         ['x', 'x', 'x', 'x', 'x', 'x', 'b', 'b'],
#         ['b', 'b', 'x', 'x', 'x', 'x', 'x', 'x'],
#         log_obj=log
#     )

# r = generate_optimized_equation_with_or(TT)
# log.println(f"{r}")
# r = generate_optimized_equation_with_and(TT)
# log.println(f"{r}")

# exit()


if False:
    # finding the best patterns for each faulty transistor
    bit_len = 8
    MULTI_COMPUTER = True
    MAX_COMPUTER = 3
    CURRENT_COMPUTER = 0

    log = Log(f"{__file__}.{CURRENT_COMPUTER}.log")


    if MULTI_COMPUTER == False:
        fa_i_range = range(bit_len-1)
    else:
        fa_i_range = range(bit_len-1)
        chunk_size = math.ceil(len(fa_i_range)/MAX_COMPUTER)
        partition = [fa_i_range[i:i+chunk_size] for i in range(0, len(fa_i_range), chunk_size)]
        
        fa_i_range = partition[CURRENT_COMPUTER]

    for fa_i in fa_i_range:
        for fa_j in range(bit_len):
            for t_index in range(6):
                _start_time = time.time()

                data_set_log_name = f"dataset/fa_i-{fa_i}-fa_j-{fa_j}-t_index-{t_index}.txt"
                log.println(data_set_log_name)
                input_data = load_pattern_file(data_set_log_name)

                # keep the accuracy +89%
                MIN_ACCURACY = 0.89
                MAX_BIT = 5
                max_bit = 0
                max_accuracy = 0
                while (max_accuracy < MIN_ACCURACY) and (max_bit < MAX_BIT):
                    max_bit += 1
                    
                    r = multi_process_best_pattern_finder(
                        input_data,
                        # range(max_bit),
                        # range(max_bit),
                        max_bit,
                        log_obj=False,
                        max_process_count=100
                    )
                    max_accuracy = r['accu']

                    log.println(f"{r}")
                    log.println(f"max bit:\t{max_bit}")
                    log.println(f"accuracy:\t{r['accu']}\t+{MIN_ACCURACY}[{'TRUE' if max_accuracy>MIN_ACCURACY else 'FALSE'}]")
                

                _end_time = time.time()
                _exe_time = _end_time - _start_time

                log.println("")
                # log.println(f"{r}")
                # log.println(f"max bit:\t{max_bit}")
                # log.println(f"accuracy:\t{r['accu']}\t+{MIN_ACCURACY}[{'TRUE' if max_accuracy>MIN_ACCURACY else 'FALSE'}]")
                # log.println(f"exe time:\t{_exe_time:.2f}s \n")
                










if True:
    # generate logical equation for each faulty transistor using the best pattern log file
    
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


    def convert_logical_expression(expression):
        """Convert bitwise operators to Python logical operators."""
        return (expression
                .replace("&", "and")
                .replace("|", "or")
                .replace("~", "not "))

    optimizer_equation = [""]
    def eq_optimizer_accept(neg_mp: MPn_v3, bin_A, bin_B):
        "normal aging equation"
        "(B0 & ~A6) | (B0 & ~A7 & ~B1) | (A6 & B1 & ~A7 & ~B0)"

        global optimizer_equation
        # eq = "(B0 & ~A6) | (B0 & ~A7 & ~B1) | (A6 & B1 & ~A7 & ~B0)"
        
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
    
    def normal_eq_optimizer_accept(neg_mp: MPn_v3, bin_A, bin_B):
        # eq = "(B0 & ~A6) | (B0 & ~A7 & ~B1) | (A6 & B1 & ~A7 & ~B0)"
        eq = "(A6 | B0) & (B0 | B1) & (~A6 | ~A7) & (~A6 | ~B0 | ~B1)"
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
        
    
    def parse_file(file_path):
        results = []
        
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Regex to extract dataset info
        dataset_pattern = r"fa_i-(\d+)-fa_j-(\d+)-t_index-(\d+)"
        # Regex to extract max bit
        max_bit_pattern = r"max bit:\s+(\d+)"
        # Extract patterns A and B
        pattern_dict_pattern = r"{.*}"

        fa_i, fa_j, t_index = None, None, None

        for i, line in enumerate(lines):
            if "dataset" in line:
                match = re.search(dataset_pattern, line)
                if match:
                    fa_i, fa_j, t_index = match.groups()

            elif "{" in line:
                pattern_match = re.search(pattern_dict_pattern, line)
                if pattern_match:
                    pattern_dict = ast.literal_eval(pattern_match.group())  # Safely parse dictionary
                    A = pattern_dict.get('A')
                    B = pattern_dict.get('B')

            elif "max bit" in line:
                max_bit_match = re.search(max_bit_pattern, line)
                if max_bit_match:
                    max_bit = int(max_bit_match.group(1))
                    # Store the extracted data
                    results.append({
                        'fa_i': int(fa_i),
                        'fa_j': int(fa_j),
                        't_index': int(t_index),
                        'max_bit': int(max_bit),
                        'A': A,
                        'B': B,
                    })

        return results

    log = Log(f"{__file__}.faulty_transistor_equation.{1}.log")    
    filename = f'{__file__}.{"faulty_transistor_best_pattern"}.log'
    parsed_data = parse_file(filename)

    equation_list = []
    current_transistor = (-1, 0, 0)
    
    for entry in parsed_data:
        fa_i, fa_j, t_index = entry['fa_i'], entry['fa_j'], entry['t_index']
        max_bit = entry['max_bit']
        A_bit_pattern, B_bit_pattern = entry['A'], entry['B']
        faulty_transistor = {'fa_i': fa_i, 'fa_j': fa_j, 't_index': t_index, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
        
        # if not(fa_i==3 and fa_j==4 and t_index==3):
        #     continue
        
        if (fa_i <= 3):
            # skip
            continue
        
        if (fa_i, fa_j, t_index) != current_transistor:
            log.println("\n")
        log.println(f"transistor: {(fa_i, fa_j, t_index)} >>> {A_bit_pattern} {B_bit_pattern}")
        input_data = load_pattern_file(f"dataset/fa_i-{fa_i}-fa_j-{fa_j}-t_index-{t_index}.txt")
                
        if current_transistor != (fa_i, fa_j, t_index):
        # if False:
            # run normal aging and normal optimized aging to compare
            
            # normal aging lifetime
            alpha_lst = MultiplierStressTest(8, None, None, optimizer_enable=False).run(log_obj=False)
            fail_transistor = get_life_expect(alpha_lst, 8, faulty_transistor)
            unoptimized_lifetime = fail_transistor["t_week"]
            log.println(f"unoptimized lifetime: [{unoptimized_lifetime} weeks] ({fail_transistor})")
            
            # dataset lifetime
            alpha_lst = MultiplierStressTest(8, dataset_optimizer_trigger, dataset_optimizer_accept).run(log_obj=False)
            fail_transistor = get_life_expect(alpha_lst, 8, faulty_transistor)
            dataset_lifetime = fail_transistor["t_week"]
            log.println(f"dataset lifetime: [{dataset_lifetime} weeks] ({fail_transistor})")
            
            # normal equation lifetime
            alpha_lst = MultiplierStressTest(8, eq_optimizer_trigger, normal_eq_optimizer_accept).run(log_obj=False)
            fail_transistor = get_life_expect(alpha_lst, 8, faulty_transistor)
            normal_eq_lifetime = fail_transistor["t_week"]
            log.println(f"normal EQ lifetime: [{normal_eq_lifetime} weeks] ({fail_transistor})")

        TT = truth_table(
                input_data=input_data,
                A_bit_pattern=A_bit_pattern,
                B_bit_pattern=B_bit_pattern,
                log_obj=False
            )
        
        eq = generate_optimized_equation_with_or(TT)
        equation_list.append(eq)
        
        
        optimizer_equation[0] = eq
        alpha_lst = MultiplierStressTest(8, eq_optimizer_trigger, eq_optimizer_accept).run(log_obj=False)
        fail_transistor = get_life_expect(alpha_lst, 8, faulty_transistor=faulty_transistor)
        eq_lifetime = fail_transistor["t_week"]
        # eq_lifetime = -1
        
        log.println(f"{max_bit} >>> equation: {eq} [{eq_lifetime} weeks] ({fail_transistor})")
        
        current_transistor = (fa_i, fa_j, t_index)

        
    #histogram of equations
    # equation_count = Counter(equation_list)
    # unique_count = len(equation_count)

    # log.println(f"Number of unique equation: {unique_count}")
    # log.println("Occurrences of each equation:")
    # for eq, count in equation_count.items():
    #     log.println(f"{eq}: {count}")
