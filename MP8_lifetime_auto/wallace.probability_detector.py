
from itertools import *
import multiprocessing
import re
import itertools
import time
import math
from tool.log import Log, Progress
from sympy import symbols, Or, And, Not, simplify_logic
from pyeda.inter import expr
from collections import Counter
import ast
from wallace_lifetime import wallace_alpha
from msimulator.Multiplier import Wallace_comp
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


def calc_accuracy(input_data, A_bit_pattern=False, B_bit_pattern=False, log_obj=False, TT=False):
    ML_table = TT
    if not TT:
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


def calc_f1_score(TT, log_obj=False):
    ML_table = TT

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

    precision = true_positive / ((true_positive + false_positive) or 1)
    recall = true_positive / ((true_positive + false_negative) or 1)
    f1_score = 2 * (precision * recall) / ((precision + recall) or 1)
    return f1_score


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

    bar = Progress(bars=1)
    bar_counter = 0
    bar_max = len(A_pattern)

    for a_pattern in A_pattern:
        B_pattern = list(product(['b', 'x'], repeat=8))
        B_pattern = [list(lst) for lst in B_pattern]


        bar.update(0, bar_counter / bar_max)
        bar_counter += 1

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


# =================================

optimizer_equation = [""]
def eq_optimizer_trigger(mp):
    global optimizer_equation
    
    eq = optimizer_equation[0]
    logical_expression = convert_logical_expression(eq)

    bin_A = mp.A
    bin_B = mp.B
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

def convert_logical_expression(expression):
    """Convert bitwise operators to Python logical operators."""
    return (expression
            .replace("&", "and")
            .replace("|", "or")
            .replace("~", "not "))

def eq_optimizer_accept(neg_mp):
    return True


# =========================================================================================================
# =========================================================================================================
# =========================================================================================================

if False:
    # healthy optimizer, find the best bit pattern with showing accuracy and TP params
    log = Log(f"{__file__}.log")
    PROCESS_COUNT = 25
    START_BIT = 8
    END_BIT = 10 + 1

    # log_filepath = 'pattern.txt'
    log_filepath = "pattern-wallace8.txt"
    input_data = load_pattern_file(log_filepath)

    count = 0
    for data in input_data:
        count += 1 if data[2] else 0
    log.println(f"total number of TRUE patterns: {count} \n")



    for bit_length in range(START_BIT, END_BIT):
        _start_time = time.time()
        
        r = multi_process_best_pattern_finder(
                input_data,
                count_bit_pattern=bit_length,
                max_process_count=PROCESS_COUNT,
                log_obj=False
            )
        log.println(f"equation bit length: {bit_length}")
        log.println(f"{r}")
        
        _end_time = time.time()
        log.println(f"time: \t{_end_time - _start_time} s | {PROCESS_COUNT} processes")

        TT = truth_table(
            input_data=input_data,
            A_bit_pattern=r['A'],
            B_bit_pattern=r['B'],
            log_obj=False
        )

        eq = generate_optimized_equation_with_or(TT)
        log.println(f"equation: \t{eq}")

        f1_score = calc_f1_score(TT, log_obj=False)
        log.println(f"F1 score: {f1_score}")

        optimizer_equation[0] = eq
        alpha_lst = wallace_alpha(Wallace_comp, 8, eq_optimizer_trigger, eq_optimizer_accept, op_enable=True, log_obj=False)

        fail_transistor = get_life_expect(alpha_lst, 8, faulty_transistor=False)
        log.println(f"fail_transistor: \t{fail_transistor} \n\n")



if True:
    # process variation lifetime using the normal selector equation
    BIT_LEN = 8
    log = Log(f"{__file__}.log")


    optimizer_equation[0] = "(B0 & ~A6 & ~B1) | (B1 & ~A6 & ~B0) | (A6 & B6 & ~A7 & ~B7) | (A6 & B7 & ~A7 & ~B6)"
    unoptimized_alpha = wallace_alpha(Wallace_comp, 8, None, None, op_enable=False)
    optimized_alpha = wallace_alpha(Wallace_comp, BIT_LEN, eq_optimizer_trigger, eq_optimizer_accept, op_enable=True)

    for fa_i in range(BIT_LEN - 1):
        for fa_j in range(BIT_LEN):
            for t_index in range(6):
                faulty_transistor = {'fa_i': fa_i, 'fa_j': fa_j, 't_index': t_index, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
                unoptimized_lifetime = get_life_expect(unoptimized_alpha, BIT_LEN, faulty_transistor)["t_week"]
                eq_lifetime = get_life_expect(optimized_alpha, BIT_LEN, faulty_transistor)["t_week"]

                log.println(f"{unoptimized_lifetime:03} -> {eq_lifetime:03}")



