"""
NOTE:


max delay: 823.161 p Seconds





"""

import re

def filter_data(data):
    pattern = r"(\w+)\s+([\d.]+)p"
    matches = re.findall(pattern, data)

    specified_values = ['o7', 'cout']
    filtered_matches = [float(match[1]) for match in matches if match[0] in specified_values]


    return filtered_matches



max_delay = 0

for A in range(2**8):
    for B in range(2**8):
        try:
            log_file_name = f"raw_data_ADD8/log-ADD8-delay-{A}-{B}.txt"

            data = open(log_file_name).read()
            result = filter_data(data)
            print(f"{A}+{B}\tdelay: {result}")

            max_delay = max(max_delay, *result)
        except:
            pass

print(f"max delay: {max_delay}")


