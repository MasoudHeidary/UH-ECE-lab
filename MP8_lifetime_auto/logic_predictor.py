import numpy as np
from pyeda.inter import *
from itertools import *
import re

def parse_log_line(line):
    pattern = r"\[\w+ \w+ \d+ \d+:\d+:\d+ \d+\] >> \[(.*?)\], \[(.*?)\], \[compliment: (True|False)\]"
    match = re.search(pattern, line)
    
    if match:
        A = list(map(int, match.group(1).split(", ")))
        B = list(map(int, match.group(2).split(", ")))
        result = True if match.group(3) == "True" else False
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


log_filepath = 'pattern.txt'
# log_filepath = 'MP8_lifetime_auto/pattern.txt'
input_data = load_log_file(log_filepath)

A_vars = exprvars('A', 8)
B_vars = exprvars('B', 8)
all_vars = A_vars + B_vars

# Step 2: Create the truth table
rows = []
outputs = []

for A, B, result in input_data:
    combined = A + B
    rows.append(tuple(combined))
    # outputs.append(1 if result else 0)
    outputs.append(result)


_counter = 0
for i in range(0, len(outputs)):
    if outputs[i] == True:
        _counter += 1

        if _counter % 2 == 0:
            _counter = 0
            outputs[i] = '-'



tt = truthtable(all_vars, outputs)

# simplified_expr = truthtable2expr(tt)
# print("Simplified Boolean Expression:")
# print(len(str(simplified_expr)))


dnf_expr = truthtable2expr(tt)
simplified_exprs = espresso_exprs(dnf_expr)
for expr in simplified_exprs:
    print("Simplified Expression:")
    # print(expr)
    print(len(str(expr)))


# x = espresso_tts(tt)
# print(len(str(x)))