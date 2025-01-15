"""
NOTE:

max delay: 36.3284ns






"""

import re

def filter_data(data):
    pattern = r"(\w+)\s+([\d.]+)p"
    matches = re.findall(pattern, data)

    specified_values = ['out_delay']
    filtered_matches = [float(match[1]) for match in matches if match[0] in specified_values]


    return filtered_matches



max_delay = 0

for i in range(0, 2**4):

    log_file_name = f"raw_data_predictor/logic_predictor_{i}.txt"

    try:
        data = open(log_file_name).read()
        result = filter_data(data)
        print(f"{i} output delay: {result}")

        max_delay = max(max_delay, *result)

    except:
        pass

print(f"max delay: {max_delay}")

