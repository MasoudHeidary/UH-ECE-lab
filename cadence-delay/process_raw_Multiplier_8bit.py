"""
NOTE:

max delay: 1.94212ns






"""




import re
import matplotlib.pyplot as plt 

def filter_data(data):
    pattern = r"(\d+)\s+([\d.]+)n"
    matches = re.findall(pattern, data)

    specified_values = ['14', '15']
    filtered_matches = [float(match[1]) for match in matches if match[0] in specified_values]

    return filtered_matches


max_delay = 0

for A in range(256):
    for B in range(256):

        log_name = f"raw_data_Multiplier_8bit/log-MP8-delay-{A}-{B}.txt"
        try:
            data = open(log_name).read()
            
            result = filter_data(data)
            print(f"{A} {B}: {result}")
        
            max_delay = max(max_delay, *result)
        except:
            pass
        
print(f"max delay: {max_delay}ns")

