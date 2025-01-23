

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
from random import randint
import pickle


log_filename = f"{__file__}.log"
log = Log(log_filename)
eq_log = Log(f"{__file__}.equations.log")

transistor_list = [
    #(transistor_location, normal_eq_lifetime, dataset_lifetime)
    ((2, 7, 1),     63,     137 ),
    ((0, 6, 1),     51,     112 ),
    ((0, 7, 2),     68,     112 ),
    ((0, 7, 5),     68,     112 ),
    ((1, 7, 2),     51,     112 ),
    ((1, 7, 5),     51,     112 ),
    ((2, 7, 2),     51,     112 ),
    ((2, 7, 5),     51,     112 ),
    ((3, 7, 2),     51,     112 ),
    ((3, 7, 5),     51,     112 ),
    ((4, 7, 2),     51,     112 ),
    ((4, 7, 5),     51,     112 ),
    ((5, 7, 2),     51,     112 ),
    ((5, 7, 5),     51,     112 ),
    ((6, 7, 3),     51,     111 ),
    ((6, 7, 4),     51,     111 ),
    ((1, 6, 2),     112,    190 ),
    ((1, 6, 5),     112,    190 ),
    ((1, 5, 1),     62,     169 ),
    ((2, 6, 2),     108,    189 ),
    ((2, 6, 5),     108,    189 ),
    ((1, 3, 0),     68,     162 ),
    ((2, 6, 0),     83,     146 ),
    ((1, 4, 0),     112,    160 ),
    ((0, 3, 0),      38,    89  ),
    ((0, 4, 0),      38,    89  ),
    ((3, 6, 2),     108,    189 ),
    ((3, 6, 5),     108,    189 ),
    ((1, 2, 0),     68,     158 ),
    ((0, 1, 0),      38,    88  ),
    ((4, 6, 2),     107,    189 ),
    ((4, 6, 5),     107,    189 ),
    ((5, 6, 2),     107,    189 ),
    ((5, 6, 5),     107,    189 ),
    ((6, 6, 3),     106,    189 ),
    ((6, 6, 4),     106,    189 ),
    ((0, 2, 0),      38,    85  ),
    ((2, 5, 0),     86,     162 ),
    ((3, 5, 0),     100,    175 ),
    ((0, 6, 3),     68,     143 ),
    ((0, 6, 4),     68,     143 ),
    ((3, 6, 0),     111,    161 ),
    ((2, 5, 2),     112,    187 ),
    ((2, 5, 5),     112,    187 ),
    ((0, 5, 2),     98,     141 ),
    ((0, 5, 5),     98,     141 ),
    ((2, 3, 0),     119,    181 ),
    ((1, 6, 0),      62,    96  ),
    ((0, 4, 2),     68,     138 ),
    ((0, 4, 5),     68,     138 ),
    ((1, 1, 0),     68,     137 ),
    ((6, 5, 3),     113,    189 ),
    ((6, 5, 4),     113,    189 ),
    ((0, 3, 2),     68,     135 ),
    ((0, 3, 5),     68,     135 ),
    ((1, 4, 2),     111,    174 ),
    ((1, 4, 5),     111,    174 ),
    ((2, 4, 1),     82,     172 ),
    ((4, 5, 0),     120,    179 ),
    ((1, 5, 3),     88,     169 ),
    ((1, 5, 4),     88,     169 ),
    ((3, 4, 0),     100,    171 ),
    ((5, 5, 0),     134,    184 ),
    ((2, 2, 0),     88,     163 ),
    ((1, 0, 0),     68,     125 ),
    ((4, 4, 0),     108,    176 ),
    ((2, 1, 0),     88,     161 ),
    ((1, 3, 2),     88,     160 ),
    ((1, 3, 5),     88,     160 ),
    ((2, 0, 0),     88,     160 ),
    ((3, 4, 2),     116,    178 ),
    ((3, 4, 5),     116,    178 ),
    ((0, 2, 2),     68,     123 ),
    ((0, 2, 5),     68,     123 ),
    ((0, 0, 0),      38,    68  ),
    ((0, 0, 0),      38,    68  ),
    ((5, 3, 0),     111,    172 ),
    ((2, 4, 3),     100,    171 ),
    ((2, 4, 4),     100,    171 ),
    ((3, 3, 1),     95,     168 ),
    ((4, 3, 0),     107,    168 ),
    ((3, 3, 3),     105,    165 ),
    ((3, 3, 4),     105,    165 ),
    ((4, 2, 1),     103,    164 ),
    ((3, 1, 0),     99,     162 ),
    ((3, 0, 0),     99,     160 ),
    ((5, 1, 0),     118,    158 ),
    ((3, 3, 0),     133,    154 ),
    ((2, 4, 2),     127,    148 ),
    ((2, 4, 5),     127,    148 ),
    ((0, 1, 2),     68,     101 ),
    ((0, 1, 5),     68,     101 ),
    ((4, 6, 1),     94,     104 ),
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

# === cache
class CACHE():
    def __init__(self, MAX_LENGTH=10000, filename=None):
        self.max_length = MAX_LENGTH
        self.filename = filename
        self.key_list = []
        self.value_list = []
        
    
    def hit_cache(self, key):
        return (key in self.key_list)
    
    def get_cache(self, key):
        if not self.hit_cache(key):
            raise LookupError("no cache hit")
        
        return self.value_list[self.key_list.index(key)]
    
    def add_cache(self, key, value=None):
        # overflow protection, remove a random value from cache
        if len(self.key_list) >= self.max_length:
            rnd_index = randint(0, self.max_length-1)
            self.key_list.pop(rnd_index)
            self.value_list.pop(rnd_index)

            log.println(f"CACHE WARNING: overflow max [{self.max_length}]")

        self.key_list.append(key)
        self.value_list.append(value)

        if self.filename:
            self.dump_cache()

    def reset_cache(self):
        self.key_list = []
        self.value_list = []

    def dump_cache(self):
        if not self.filename:
            raise RuntimeError("in-memory cache can not be saved")
        
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)
        
    @classmethod
    def load_cache(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


# === cache END

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



"""
NOTE:
progressive method, we will get the set of equations and pace transistors one by one 
and drop the equations that are not compatible
"""
if False:
    # generate all equations possible, and calculate how much lifeitme each get
    BIT_COUNT = 4
    bit_pattern_list = generate_all_bit_pattern(BIT_COUNT)
    log.println(f"total bit pattern: {len(bit_pattern_list)}, bit count: {BIT_COUNT}")

    ### CONFIG
    initial_transistor = transistor_list[0]
    group_transistor = transistor_list[1:]


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

            # the new equation should be 20% better than normal equation at least
            if (eq_lifetime >= normal_eq_lifetime * 1.2) or (eq_lifetime >= dataset_lifetime):
                description = "New EQ Saved"
                equation_list.append(equation)
            else:
                description = "New EQ Regarded"
                false_equation_list.append(equation)


        # log.println(f"{bit_pattern} \t>>> {equation} \t>>> {eq_lifetime} [{normal_eq_lifetime}, {dataset_lifetime}] \t[{description}]")
        log.println(f"{"..."} \t>>> {equation:60} \t>>> {eq_lifetime} [{normal_eq_lifetime}, {dataset_lifetime}] \t[{description}]")
    log.println(f"bit pattern process DONE")


    for t in group_transistor:
        fa_i, fa_j, t_index = t[0]
        faulty_transistor = {'fa_i': fa_i, 'fa_j': fa_j, 't_index': t_index, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
        normal_eq_lifetime = t[1]
        dataset_lifetime = t[2]
        
        loop_equation_list = equation_list[:]
        loop_equation_list = list(set(loop_equation_list))
        equation_list = []

        log.println(f"GROUPING PROCESS")
        log.println(f"transistor: {(fa_i, fa_j, t_index)}, loop equation count: {len(loop_equation_list)}")
        log.println('equations:\n' + '\n'.join(loop_equation_list))

        # input_data = load_pattern_file(f"dataset/fa_i-{fa_i}-fa_j-{fa_j}-t_index-{t_index}.txt")

        for equation in loop_equation_list:
            optimizer_equation[0] = equation

            alpha_lst = MultiplierStressTest(8, eq_optimizer_trigger, eq_optimizer_accept).run(log_obj=False)
            fail_transistor = get_life_expect(alpha_lst, 8, faulty_transistor)
            eq_lifetime = fail_transistor["t_week"]

            description = str()
            # the new equation should be 20% better than normal equation at least
            if (eq_lifetime > normal_eq_lifetime * 1.2) or (eq_lifetime >= dataset_lifetime):
                equation_list.append(equation)
                description = "EQ OK"
            else:
                description = "EQ DROPEED"


            log.println(f"{equation:60} \t>>> {eq_lifetime} [{normal_eq_lifetime}, {dataset_lifetime}] \t{description}")
        log.println(f"transistor {(fa_i, fa_j, t_index)} DONE")
        log.println(f"\n")


# GROUPING MOST COMPATIBLE ONES
"""
NOTE:
instead of dropping the incompatible equations, we will drop the incompatible transistors
"""
if True:
    equation_cache = CACHE()
    
    alpha_cache_filename = "alpha.cache"
    try:
        alpha_lst_cache = CACHE.load_cache(alpha_cache_filename)
    except:
        alpha_lst_cache = CACHE(filename=alpha_cache_filename)
    
    # TODO: delete the range number
    for initial_transistor in transistor_list[34+1:78]:
        
        EQ_DIFF_NORMAL_EQ = 1.1     # +10%
        # note: the false_equation_list is designed for caching prupose to skip recomputing the repeating equations
        equation_list = []

        # generate all equations possible, and calculate how much lifeitme each get
        BIT_COUNT = 4
        bit_pattern_list = generate_all_bit_pattern(BIT_COUNT)

        log.println(f"initial transistor: {initial_transistor[0]}")
        log.println(f"total bit pattern: {len(bit_pattern_list)}, bit count: {BIT_COUNT}")


        # generate the initial equation set using bit_pattern_list
        fa_i, fa_j, t_index = initial_transistor[0]
        faulty_transistor = {'fa_i': fa_i, 'fa_j': fa_j, 't_index': t_index, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
        normal_eq_lifetime = initial_transistor[1]
        dataset_lifetime = initial_transistor[2]
        
        input_data = load_pattern_file(f"dataset/fa_i-{fa_i}-fa_j-{fa_j}-t_index-{t_index}.txt")
        log.println(f"transistor: {(fa_i, fa_j, t_index)}")

        for bit_pattern in bit_pattern_list:
            TT = truth_table(input_data, bit_pattern[0], bit_pattern[1], log_obj=False)
            equation = generate_optimized_equation_with_or(TT)


            description = str()
            eq_lifetime = int()
            if (equation == "0"):
                description = "NO EQ"
            
            elif equation_cache.hit_cache(equation):
                description = "EQ in Cache"

            else:
                equation_cache.add_cache(equation)

                if not alpha_lst_cache.hit_cache(equation):
                    optimizer_equation[0] = equation
                    alpha_lst = MultiplierStressTest(8, eq_optimizer_trigger, eq_optimizer_accept).run(log_obj=False)
                    alpha_lst_cache.add_cache(equation, alpha_lst)
                else:
                    alpha_lst = alpha_lst_cache.get_cache(equation)

                fail_transistor = get_life_expect(alpha_lst, 8, faulty_transistor)
                eq_lifetime = fail_transistor["t_week"]

                # the new equation should be 20% better than normal equation at least
                if (eq_lifetime >= normal_eq_lifetime * 1.2) or (eq_lifetime >= dataset_lifetime):
                    description = "New EQ Saved"
                    equation_list.append(equation)
                else:
                    description = "New EQ Regarded"


            # log.println(f"{bit_pattern} \t>>> {equation} \t>>> {eq_lifetime} [{normal_eq_lifetime}, {dataset_lifetime}] \t[{description}]")
            eq_log.println(f"{"..."} \t>>> {equation:60} \t>>> {eq_lifetime} [{normal_eq_lifetime}, {dataset_lifetime}] \t[{description}]")
        log.println(f"bit pattern process DONE")




        log.println(f"GROUPING PROCESS >>> initial transistor \t {(fa_i, fa_j, t_index)}")
        for eq in equation_list:
            
            if alpha_lst_cache.hit_cache(eq):
                alpha_lst = alpha_lst_cache.get_cache(eq)
            else:
                optimizer_equation[0] = eq
                alpha_lst = MultiplierStressTest(8, eq_optimizer_trigger, eq_optimizer_accept).run(log_obj=False)
                log.println(f"CACHE ERROR: alpha list MISS")

            added_t_list = []
            for t in transistor_list:
                fa_i, fa_j, t_index = t[0]
                faulty_transistor = {'fa_i': fa_i, 'fa_j': fa_j, 't_index': t_index, 'x_vth_base': 1.1, 'x_vth_growth': 1.1}
                normal_eq_lifetime = t[1]
                dataset_lifetime = t[2]

                fail_transistor = get_life_expect(alpha_lst, 8, faulty_transistor)
                eq_lifetime = fail_transistor["t_week"]

                description = str()
                if(eq_lifetime > normal_eq_lifetime * EQ_DIFF_NORMAL_EQ) or (eq_lifetime >= dataset_lifetime):
                    description = "TRANSISTOR ADDED"
                    added_t_list.append([t, eq_lifetime])
                else:
                    description = "TRANSISTOR SKIPPED"
                
                # log.println(f"{(fa_i, fa_j, t_index)} \t>>> {eq_lifetime} > {normal_eq_lifetime} \t>>> {description}")

            log.println(f"{eq} >>> works for {len(added_t_list):3}/{len(transistor_list):3}")
            log_str = "transistor list: "
            for t in added_t_list:
                log_str += f"\n [{transistor_list.index(t[0])}] {t[0][0]} \t {t[1]} <- {t[0][1]}"
            log.println(log_str)



"""
NOTE:
choosing the best equations based on how many equations is needed
"""
if False:
    
    def parse_equation_file(filename):
        data = []

        with open(filename, 'r') as file:
            lines = file.readlines()

        current_equation = None
        current_transistors = []

        for line in lines:
            equation_match = re.match(r".*>>\s+([^>]+)>>> works for.*", line)
            if equation_match:
                if current_equation and current_transistors:
                    data.append({"equation": current_equation, "transistors": current_transistors})

                current_equation = equation_match.group(1)
                current_transistors = []
            
            transistor_match = re.match(r"\s*\[\d+]\s*\((\d+, \d+, \d+)\)", line)
            if transistor_match:
                transistor_location = tuple(map(int, transistor_match.group(1).split(", ")))
                current_transistors.append(transistor_location)

        # if current_equation and current_transistors:
        #     data.append({"equation": current_equation, "transistors": current_equation})

        return data
    

    equation_file = parse_equation_file(log_filename)
    equation_list = [i["equation"] for i in equation_file]
    transistor_list = [i["transistors"] for i in equation_file]

    def get_best_equations(eq_count_needed):
        r_eq = []
        r_t = []

        for _ in range(eq_count_needed):
            max_t_lst = []
            max_eq = None
            for i, eq in enumerate(equation_list):
                next_t_lst = list(set(r_t + transistor_list[i]))

                if len(next_t_lst) > len(max_t_lst):
                    max_t_lst = next_t_lst.copy()
                    max_eq = eq
            
            r_eq.append(max_eq)
            r_t = max_t_lst.copy()
    
        return (r_eq, r_t)
        

            
    
    COUNT_EQ = 2
    top_eq = get_best_equations(COUNT_EQ)
    print(top_eq[0])
    print(top_eq[1])
    print(f"covered: {len(top_eq[1])}")

    # for i in range(len(equation_list)):
    #     print(equation_list[i])
    #     print(transistor_list[i])
    #     print('-'*40)





    

