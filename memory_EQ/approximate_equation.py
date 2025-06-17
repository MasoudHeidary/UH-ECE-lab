
"""
deisgned for 15bit data
"""

import itertools
from tool.log import Log
import re
from sympy import symbols, Or, And, Not, simplify_logic
from random import randint
import multiprocessing
import time


#CONFIG
MAX_PROCESSES = multiprocessing.cpu_count()
# MAX_PROCESSES = 20
log = Log(f"{__file__}.log", terminal=True)




def parse_pattern_line(line):
    pattern = r"\[(.*?)\], \[output: (\d)\]"
    match = re.search(pattern, line.strip())

    if match:
        A = list(map(int, match.group(1).split(", ")))
        result = int(match.group(2))
        return A, result
    return None


def load_pattern_file(filepath):
    input_data = []
    
    with open(filepath, 'r') as file:
        for line in file:
            parsed_data = parse_pattern_line(line)
            if parsed_data:
                input_data.append(parsed_data)
    
    return input_data


def comp(x_data, solid_data):
    if len(x_data) != len(solid_data):
        raise RuntimeError()
    
    for i, value in enumerate(x_data):
        if value == 'x':
            continue
        if value != solid_data[i]:
            return False
    
    return True

def calc_prob(input_data, partial_A):
    false_count = 0
    true_count = 0

    for data in input_data:
        A = data[0]
        out = data[1]

        if comp(partial_A, A) and (out == True):
            true_count += 1
        elif comp(partial_A, A) and (out == False):
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


def truth_table(input_data, A_bit_pattern, log_obj=False):
    if log_obj:
        log.println("Truth Table >> Training")
    A_patterns = generate_patterns(A_bit_pattern)

    ML_table = []
    for a in A_patterns:
            true_prob, false_prob = calc_prob(input_data, a)
            if log_obj:
                log_obj.println(f"{a}: {true_prob, false_prob} {true_prob/(false_prob or 0.001):.1f} [{'True' if true_prob > false_prob else 'False'}]")

            ML_table.append(
                {'A': a, 'out': true_prob > false_prob}
            )

    return ML_table

# def generate_equation(truth_table):
#     terms = []
#     variables = [
#         'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
#         'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14'
#         ]
#     for row in truth_table:
#         inputs, output = row['A'], row['out']
#         if output == 1:
#             term = []
#             for var, val in zip(variables, inputs):
#                 if val != 'x':  # Ignore 'x' bits
#                     term.append(f"{var}" if val == 1 else f"~{var}")
#             terms.append(" & ".join(term))

#     equation = " | ".join(f"({term})" for term in terms)
#     return equation if equation else "0"


def generate_optimized_equation_with_or(truth_table):
    variables = symbols('A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14')
    minterms = []
    for row in truth_table:
        inputs, output = row['A'], row['out']
        if output == 1:
            term = []
            for var, val in zip(variables, inputs):
                if val != 'x':
                    term.append(var if val == 1 else Not(var))
            minterms.append(And(*term))

    equation = Or(*minterms)
    optimized_equation = simplify_logic(equation, form='dnf')

    return str(optimized_equation) if optimized_equation else "0"


def interface(table, data):
    for tb in table:
        t_A = tb['A']

        if comp(t_A, data[0]):
            t_out = tb['out']
            return t_out
    raise RuntimeError("table FAILED!")


def calc_accuracy(input_data, A_bit_pattern=False, log_obj=False, TT=False):
    ML_table = TT
    if not TT:
        ML_table = truth_table(input_data, A_bit_pattern, log_obj)

    if log_obj:
        log_obj.println(f"Calc Accuracy >> testing")

    # testing
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for data in input_data:
        y_pred = interface(ML_table, data)
        d_out = data[1]

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
        log_obj.println(f"{A_bit_pattern}")
        log_obj.println(f"Accuracy: {accu: 0.3f}, TP: {true_positive}, TN: {true_negative}, FP(not suppose compliment): {false_positive}, FN(lost compliments):{false_negative},")
    return accu, true_positive, true_negative, false_positive, false_negative


def calc_f1_score(input_data, TT, log_obj=False):
    ML_table = TT

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for data in input_data:
        y_pred = interface(ML_table, data)
        d_out = data[1]

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

    def work_wrapper(input_data, a_pattern, log_obj, results):
        accu, TP, TN, FP, FN = calc_accuracy(input_data, a_pattern, log_obj=False)
        if log_obj:
            log_obj.println(f"{a_pattern}\t{accu} ({(TP, TN, FP, FN)})")
        results.append({
            'A': a_pattern,
            'accu': accu,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        })

    processes = []
    processes_count = 0
    result = multiprocessing.Manager().list()

    A_pattern = list(itertools.product(['b', 'x'], repeat=15))
    A_pattern = [list(lst) for lst in A_pattern]

    for a_pattern in A_pattern:
        if a_pattern.count('b') != count_bit_pattern:
            continue

        processes.append(
            multiprocessing.Process(
                target=work_wrapper,
                args=(input_data, a_pattern, False, result),
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
            pattern['A'].count('b')
            ) < (
            best_accuracy_pattern['A'].count('b')
            ):
            best_accuracy_pattern = pattern
    
    if log_obj:
        log_obj.println(f"**max accuracy: {max_accuracy}")
        log_obj.println(f"**processes count: {processes_count}")

        _end_time = time.time()
        _func_time = _end_time - _start_time
        log.println(f"**execution time:\t{_func_time:.2f}s")
    return best_accuracy_pattern





if __name__ == "__main__":
    TT_filepath = 'bit_representation_1.txt'
    TT_data = load_pattern_file(TT_filepath)
    log.println("TT input data loaded")

    for pattern_len in range(10, 15):
        log.println(f"bit equation length {pattern_len} RUNNING")

        r = multi_process_best_pattern_finder(
            TT_data,
            count_bit_pattern = pattern_len,
            max_process_count = MAX_PROCESSES,
            log_obj=log,
        )
        log.println(f"best pattern: {r}")

        TT = truth_table(
            TT_data,
            A_bit_pattern = r['A'],
            log_obj=False,
        )
        
        eq = generate_optimized_equation_with_or(TT)
        log.println(f"equation: \n{eq}")
        
        f1 = calc_f1_score(
            TT_data,
            TT,
            log_obj=False
        )
        log.println(f"F1 score: {f1}")

        log.println(f"\n{'='*50}")

