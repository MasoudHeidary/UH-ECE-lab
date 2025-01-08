

"""
purpose:
the list of critical transistors that need special optimizer and try to find a equation to be useable by a group of them
"""

from itertools import *
import itertools
from tool.log import Log
from msimulator.Multiplier import MPn_v3
import re
from sympy import symbols, Or, And, Not, simplify_logic
from msimulator.get_alpha_class import MultiplierStressTest
from get_life_expect import get_life_expect


log = Log(f"{__file__}.log")

transistor_list = [
    #(transistor_location, normal_eq_lifetime, dataset_lifetime)
    ((2, 7, 1), 63,     137 ),
    ((0, 6, 1), 51,     112 ),
    ((0, 7, 2), 68,     112 ),
    ((0, 7, 5), 68,     112 ),
    ((1, 7, 2), 51,     112 ),
    ((1, 7, 5), 51,     112 ),
    ((2, 7, 2), 51,     112 ),
    ((2, 7, 5), 51,     112 ),
    ((3, 7, 2), 51,     112 ),
    ((3, 7, 5), 51,     112 ),
    ((4, 7, 2), 51,     112 ),
    ((4, 7, 5), 51,     112 ),
    ((5, 7, 2), 51,     112 ),
    ((5, 7, 5), 51,     112 ),
    ((6, 7, 3), 51,     111 ),
    ((6, 7, 4), 51,     111 ),
    ((1, 6, 2), 112,    190 ),
    ((1, 6, 5), 112,    190 ),
    ((1, 5, 1), 62,     169 ),
    ((2, 6, 2), 108,    189 ),
    ((2, 6, 5), 108,    189 ),
    ((1, 3, 0), 68,     162 ),
    ((2, 6, 0), 83,     146 ),
    ((1, 4, 0), 112,    160 ),
    ((0, 3, 0), 38,     89  ),
    ((0, 4, 0), 38,     89  ),
    ((3, 6, 2), 108,    189 ),
    ((3, 6, 5), 108,    189 ),
    ((1, 2, 0), 68,     158 ),
    ((0, 1, 0), 38,     88  ),
    ((4, 6, 2), 107,    189 ),
    ((4, 6, 5), 107,    189 ),
    ((5, 6, 2), 107,    189 ),
    ((5, 6, 5), 107,    189 ),
    ((6, 6, 3), 106,    189 ),
    ((6, 6, 4), 106,    189 ),
    ((0, 2, 0), 38,     85  ),
    ((2, 5, 0), 86,     162 ),
    ((3, 5, 0), 100,    175 ),
    ((0, 6, 3), 68,     143 ),
    ((0, 6, 4), 68,     143 ),
    ((3, 6, 0), 111,    161 ),
    ((2, 5, 2), 112,    187 ),
    ((2, 5, 5), 112,    187 ),
    ((0, 5, 2), 98,     141 ),
    ((0, 5, 5), 98,     141 ),
    ((2, 3, 0), 119,    181 ),
    ((1, 6, 0), 62,     96  ),
    ((0, 4, 2), 68,     138 ),
    ((0, 4, 5), 68,     138 ),
    ((3, 5, 2), 109,    188 ),
    ((3, 5, 5), 109,    188 ),
    ((1, 1, 0), 68,     137 ),
    ((5, 5, 2), 113,    189 ),
    ((5, 5, 5), 113,    189 ),
    ((6, 5, 3), 113,    189 ),
    ((6, 5, 4), 113,    189 ),
    ((0, 3, 2), 68,     135 ),
    ((0, 3, 5), 68,     135 ),
    ((4, 5, 2), 113,    188 ),
    ((4, 5, 5), 113,    188 ),
    ((1, 4, 2), 111,    174 ),
    ((1, 4, 5), 111,    174 ),
    ((2, 4, 1), 82,     172 ),
    ((4, 5, 0), 120,    179 ),
    ((1, 5, 3), 88,     169 ),
    ((1, 5, 4), 88,     169 ),
    ((3, 4, 0), 100,    171 ),
    ((5, 5, 0), 134,    184 ),
    ((2, 2, 0), 88,     163 ),
    ((1, 0, 0), 68,     125 ),
    ((4, 4, 0), 108,    176 ),
    ((2, 1, 0), 88,     161 ),
    ((1, 3, 2), 88,     160 ),
    ((1, 3, 5), 88,     160 ),
    ((2, 0, 0), 88,     160 ),
    ((3, 2, 0), 118,    178 ),
    ((3, 4, 2), 116,    178 ),
    ((3, 4, 5), 116,    178 ),
    ((0, 2, 2), 68,     123 ),
    ((0, 2, 5), 68,     123 ),
    ((4, 1, 0), 117,    176 ),
    ((5, 0, 0), 116,    176 ),
    ((1, 2, 2), 88,     158 ),
    ((1, 2, 5), 88,     158 ),
    ((0, 0, 0), 38,     68  ),
    ((0, 0, 0), 38,     68  ),
    ((3, 2, 2), 113,    172 ),
    ((3, 2, 5), 113,    172 ),
    ((5, 3, 0), 111,    172 ),
    ((2, 4, 3), 100,    171 ),
    ((2, 4, 4), 100,    171 ),
    ((3, 3, 1), 95,     168 ),
    ((4, 3, 0), 107,    168 ),
    ((4, 3, 2), 115,    167 ),
    ((4, 3, 5), 115,    167 ),
    ((6, 2, 0), 112,    166 ),
    ((3, 3, 3), 105,    165 ),
    ((3, 3, 4), 105,    165 ),
    ((4, 2, 1), 103,    164 ),
    ((5, 2, 0), 110,    164 ),
    ((5, 2, 2), 114,    163 ),
    ((5, 2, 5), 114,    163 ),
    ((6, 1, 0), 112,    163 ),
    ((6, 2, 3), 113,    163 ),
    ((6, 2, 4), 113,    163 ),
    ((3, 1, 0), 99,     162 ),
    ((5, 1, 1), 107,    162 ),
    ((4, 2, 3), 110,    161 ),
    ((4, 2, 4), 110,    161 ),
    ((6, 0, 1), 109,    161 ),
    ((3, 0, 0), 99,     160 ),
    ((4, 0, 0), 106,    160 ),
    ((2, 2, 2), 99,     159 ),
    ((2, 2, 5), 99,     159 ),
    ((5, 1, 0), 118,    158 ),
    ((6, 0, 0), 116,    158 ),
    ((6, 1, 3), 113,    158 ),
    ((6, 1, 4), 113,    158 ),
    ((4, 0, 1), 120,    156 ),
    ((4, 2, 0), 123,    156 ),
    ((5, 1, 3), 110,    156 ),
    ((5, 1, 4), 110,    156 ),
    ((3, 3, 0), 133,    154 ),
    ((4, 2, 2), 116,    153 ),
    ((4, 2, 5), 116,    153 ),
    ((3, 3, 2), 120,    152 ),
    ((3, 3, 5), 120,    152 ),
    ((6, 2, 2), 112,    152 ),
    ((6, 2, 5), 112,    152 ),
    ((3, 4, 1), 126,    151 ),
    ((2, 5, 1), 147,    150 ),
    ((1, 1, 2), 88,     134 ),
    ((1, 1, 5), 88,     134 ),
    ((2, 4, 2), 127,    148 ),
    ((2, 4, 5), 127,    148 ),
    ((4, 1, 3), 112,    147 ),
    ((4, 1, 4), 112,    147 ),
    ((0, 1, 2), 68,     101 ),
    ((0, 1, 5), 68,     101 ),
    ((5, 0, 1), 109,    144 ),
    ((2, 2, 3), 127,    143 ),
    ((2, 2, 4), 127,    143 ),
    ((1, 2, 3), 145,    134 ),
    ((1, 2, 4), 145,    134 ),
    ((2, 5, 3), 113,    134 ),
    ((2, 5, 4), 113,    134 ),
    ((3, 0, 1), 127,    134 ),
    ((1, 4, 3), 114,    114 ),
    ((1, 4, 4), 114,    114 ),
    ((1, 4, 1), 113,    113 ),
    ((2, 3, 1), 107,    107 ),
    ((4, 6, 1), 94,     104 ),
]

# === private old

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

def comp(x_data, solid_data):
    if len(x_data) != len(solid_data):
        raise RuntimeError()
    
    for i, value in enumerate(x_data):
        if value == 'x':
            continue
        if value != solid_data[i]:
            return False
    
    return True

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

# ===

def generate_all_bit_pattern(bit_count):
    equations = []

    A_pattern = list(product(['b', 'x'], repeat=8))
    A_pattern = [list(lst) for lst in A_pattern]
    for a_pattern in A_pattern:
        B_pattern = list(product(['b', 'x'], repeat=8))
        B_pattern = [list(lst) for lst in B_pattern]

        for b_pattern in B_pattern:
            if a_pattern.count('b') + b_pattern.count('b') == bit_count:
                equations.append([a_pattern, b_pattern])

    return equations

if True:
    # generate all equations possible, and calculate how much lifeitme each get
    BIT_COUNT = 4
    bit_pattern_list = generate_all_bit_pattern(BIT_COUNT)
    log.println(f"total bit pattern: {len(bit_pattern_list)}, bit count: {BIT_COUNT}")

    ### CONFIG
    initial_transistor = transistor_list[0]
    group_transistor = transistor_list[1:5]


    # generate the initial equation set using bit_pattern_list
    fa_i, fa_j, t_index = initial_transistor[0]
    faulty_transistor = {'fa_i': fa_i, 'fa_j': fa_j, 't_index': t_index, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
    normal_eq_lifetime = initial_transistor[1]
    dataset_lifetime = initial_transistor[2]

    input_data = load_pattern_file(f"dataset/fa_i-{fa_i}-fa_j-{fa_j}-t_index-{t_index}.txt")
    log.println(f"transistor: {(fa_i, fa_j, t_index)}")

    # note: the false_equation_list is designed for caching prupose to skip recomputing the repeating equations
    equation_list = []
    false_equation_list = []
    
    for bit_pattern in bit_pattern_list:
        TT = truth_table(input_data, bit_pattern[0], bit_pattern[1], log_obj=False)
        equation = generate_optimized_equation_with_or(TT)


        description = str()
        eq_lifetime = int()
        if (equation == "0"):
            description = "NO EQ"
        
        elif (equation in equation_list) or (equation in false_equation_list):
            description = "EQ in Cache"

        else:
            description = "EQ"
            optimizer_equation[0] = equation
            alpha_lst = MultiplierStressTest(8, eq_optimizer_trigger, eq_optimizer_accept).run(log_obj=False)
            fail_transistor = get_life_expect(alpha_lst, 8, faulty_transistor)
            eq_lifetime = fail_transistor["t_week"]

            if eq_lifetime > normal_eq_lifetime:
                description = "New EQ Saved"
                equation_list.append(equation)
            else:
                description = "New EQ Regarded"
                false_equation_list.append(equation)


        log.println(f"{bit_pattern} \t>>> {equation} \t>>> {eq_lifetime} [{normal_eq_lifetime}, {dataset_lifetime}] \t[{description}]")
    log.println(f"bit pattern process DONE")


    for t in group_transistor:
        fa_i, fa_j, t_index = t[0]
        faulty_transistor = {'fa_i': fa_i, 'fa_j': fa_j, 't_index': t_index, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
        normal_eq_lifetime = t[1]
        dataset_lifetime = t[2]
        
        loop_equation_list = equation_list.copy()
        loop_equation_list = list(set(loop_equation_list))
        equation_list = []

        log.println(f"GROUPING PROCESS")
        log.println(f"transistor: {(fa_i, fa_j, t_index)}, loop equation count: {len(loop_equation_list)}")
        log.println('\n'.join(map(str, loop_equation_list)))

        input_data = load_pattern_file(f"dataset/fa_i-{fa_i}-fa_j-{fa_j}-t_index-{t_index}.txt")


        for equation in loop_equation_list:
            optimizer_equation[0] = equation

            alpha_lst = MultiplierStressTest(8, eq_optimizer_trigger, eq_optimizer_accept).run(log_obj=False)
            fail_transistor = get_life_expect(alpha_lst, 8, faulty_transistor)
            eq_lifetime = fail_transistor["t_week"]

            if eq_lifetime > normal_eq_lifetime:
                equation_list.append(equation)


            log.println(f"{equation} \t>>> {eq_lifetime} [{normal_eq_lifetime}, {dataset_lifetime}] \t{'True' if eq_lifetime>normal_eq_lifetime else 'False'}")
        log.println("this transistor DONE \n")







            





    

