from itertools import *
import multiprocessing.process
import re
import itertools
from tool.log import Log

log = Log("probability_detector.txt")

def parse_log_line(line):
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

def load_log_file(filepath):
    input_data = []
    
    with open(filepath, 'r') as file:
        for line in file:
            parsed_data = parse_log_line(line)
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





log_filepath = 'pattern.txt'
input_data = load_log_file(log_filepath)
# print(f"data pattern: {input_data[0]}")


# if the output is True, calculate of probaility of each bit (being 1)
A_bit_sum = [0 for _ in range(8)]
B_bit_sum = [0 for _ in range(8)]
for data in input_data:
    if data[2] == True:
        for i in range(8):
            A_bit_sum[i] += data[0][i]
            B_bit_sum[i] += data[1][i]

# average_A_bit_sum = [i/(2**16) for i in A_bit_sum]
# average_B_bit_sum = [i/(2**16) for i in B_bit_sum]
# print(f"A: {average_A_bit_sum}, B: {average_B_bit_sum}")

count = 0
for data in input_data:
    count += 1 if data[2] else 0

print(f"number of TRUE patterns: {count}")

def comp(x_data, solid_data):
    if len(x_data) != len(solid_data):
        raise RuntimeError()
    
    for i, value in enumerate(x_data):
        if value == 'x':
            continue
        if value != solid_data[i]:
            return False
    
    return True


def calc_probability(partial_A, partial_B):
    global input_data
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

def calc_prob(partial_A, partial_B):
    global input_data
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

def calc_accuracy(A_bit_pattern, B_bit_pattern, LOG=False):
    global input_data

    # traing
    if LOG:
        log.println("TRAINING...")
    A_patterns = generate_patterns(A_bit_pattern)
    B_patterns = generate_patterns(B_bit_pattern)

    ML_table = []

    for a in A_patterns:
        for b in B_patterns:
            true_prob, false_prob = calc_prob(a, b)
            if LOG:
                log.println(f"{a} {b}: {true_prob, false_prob} {true_prob/(false_prob or 0.001):.1f} [{'True' if true_prob > false_prob else 'False'}]")

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
    if LOG or True:
        log.println(f"{A_bit_pattern}\t{B_bit_pattern} \t\t Accuracy: {accu: 0.3f}, TP: {true_positive}, TN: {true_negative}, FP(not suppose compliment): {false_positive}, FN(lost compliments):{false_negative},")
        # log.println(f"Accuracy: {accu: 0.3f}, TP: {true_positive}, TN: {true_negative}, FP(not suppose compliment): {false_positive}, FN(lost compliments):{false_negative}, ")
    return accu, true_positive, true_negative, false_positive, false_negative

# calc_accuracy(
#     ['b', 'b', 'x', 'x', 'x', 'x', 'b', 'b'],
#     ['b', 'b', 'x', 'x', 'x', 'x', 'b', 'b'],
# )


if False:
    A_pattern = list(product(['b', 'x'], repeat=8))
    filtered_A = [lst for lst in A_pattern if lst.count('b') <= 4]
    filtered_A = [list(lst) for lst in filtered_A]

    for a_pattern in filtered_A:
        B_pattern = list(product(['b', 'x'], repeat=8))
        filtered_B = [lst for lst in B_pattern if lst.count('b') <= 4]
        filtered_B = [list(lst) for lst in filtered_B]
        
        for b_pattern in filtered_B:
            
            # log.println(f"{a_pattern} {b_pattern}")
            calc_accuracy(a_pattern, b_pattern)

if True:
    # automatic different pattern generator
    import multiprocessing
    from itertools import product
    
    processes = []
    processes_count = 0

    A_pattern = list(product(['b', 'x'], repeat=8))
    filtered_A = [lst for lst in A_pattern if lst.count('b') <= 5]
    filtered_A = [list(lst) for lst in filtered_A]

    for a_pattern in filtered_A:
        B_pattern = list(product(['b', 'x'], repeat=8))
        filtered_B = [lst for lst in B_pattern if lst.count('b') <= 5]
        filtered_B = [list(lst) for lst in filtered_B]
        
        for b_pattern in filtered_B:
            
            processes.append(
                multiprocessing.Process(target=calc_accuracy, args=(a_pattern, b_pattern))
            )
            processes_count += 1
            
            if processes_count == 40:
                # start processes
                # log.println("Processing Batch:")
                for p in processes:
                    p.start()
                for p in processes:
                    p.join()
                    
                processes = []
                processes_count = 0
    

if False:    
    for A_msb in range(5):
        for B_msb in range(5):

            log.println()
            log.println(f"A_msb: {A_msb}, B_msb: {B_msb}")
            
            partial_A_combination = list(itertools.product([0, 1], repeat=A_msb)).copy()
            partial_B_combination = list(itertools.product([0, 1], repeat=B_msb)).copy()

            for a in partial_A_combination:
                for b in partial_B_combination:
                    a = list(a)
                    b = list(b)
                    prob = calc_probability(a, b)
                    log.println(f"{a} {b}: {prob} \t{prob[0]/prob[1]}")



if False:
    A_bit_selection = list(itertools.product([0, 1], repeat=4)).copy()
    B_bit_selection = list(itertools.product([0, 1], repeat=4)).copy()

    base = [0 for _ in range(4)] + [1 for _ in range(4)]
    unique_perm = set(itertools.permutations(base))
    valid_comb = [list(perm) for perm in unique_perm]

    for comb in valid_comb:
        print(comb)
    exit()

    for A_msb in range(4, 5):
        for B_msb in range(4, 5):

            log.println()
            log.println(f"A_msb: {A_msb}, B_msb: {B_msb}")
            
            partial_A_combination = list(itertools.product([0, 1], repeat=A_msb)).copy()
            partial_B_combination = list(itertools.product([0, 1], repeat=B_msb)).copy()

            for a in partial_A_combination:
                for b in partial_B_combination:
                    a = list(a)
                    b = list(b)
                    prob = calc_probability(a, b)
                    log.println(f"{a} {b}: {prob}")



if False:
    # lets predict and get the accuracy
    true_predict = 0
    false_predict = 0
    for data in input_data:
        A = data[0]
        B = data[1]

        if A.count(0) + B.count(0) > 12:
            if data[2] == True:
                true_predict += 1
            else:
                false_predict += 1
    print(f"true predict: {true_predict}")
    print(f"false predict: {false_predict}")
        

