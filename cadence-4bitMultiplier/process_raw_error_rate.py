import re
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter


# just modify plot fig
EXPAND_TIME_FACTOR = 2


def filter_data(data):
    pattern = r"(\d+)\s+([\d.]+)p"
    matches = re.findall(pattern, data)

    specified_values = ['7', '6', '5', '0']
    filtered_matches = [float(match[1]) for match in matches] or [0]

    return filtered_matches


def generate_input(bit_len):
    pattern = []
    for A in range(0, 2**bit_len):
        def b(num: int, length):
            return list(map(int, reversed(format(num, f'0{length}b'))))
        pattern += [b(A, bit_len)]
    return pattern

VA_lst = generate_input(4)
VB_lst = generate_input(4)


time = []
error_rate_percent = []

for t_week in range(10, 100+1, 10):

    time += [t_week]

    # normal delay and find the max delay possible
    max_normal_delay = 0
    for VA in VA_lst:
        for VB in VB_lst:
            normal_log_name = f"./raw_data_error_rate/error-rate-normal-week{t_week:03}-A{''.join(map(str, VA))}-B{''.join(map(str, VB))}.txt"

            normal_file = open(normal_log_name)
            normal_delay = max(filter_data(normal_file.read()))
            if normal_delay > max_normal_delay:
                max_normal_delay = normal_delay

    print(f"time: {t_week} weeks, max normal delay {max_normal_delay}")


    error_counter = 0
    max_modify_delay = 0
    for VA in VA_lst:
        for VB in VB_lst:
            modify_log_name = f"./raw_data_error_rate/error-rate-modified-week{t_week:03}-A{''.join(map(str, VA))}-B{''.join(map(str, VB))}.txt"

            modify_file = open(modify_log_name)
            modify_delay = max(filter_data(modify_file.read()))

            if modify_delay > max_modify_delay:
                max_modify_delay = modify_delay

            if modify_delay > max_normal_delay:
                error_counter += 1
    
    print(f"time: {t_week} weeks, max modify delay {max_modify_delay}")
    print(f"time: {t_week} weeks, error counter: {error_counter}, error_rate: {error_counter/256}")
    
    error_rate_percent.append(error_counter/256 * 100)
    

time = [t*EXPAND_TIME_FACTOR for t in time]

plt.figure(figsize=(13, 10))

plt.plot(time, error_rate_percent, label="", linewidth=5)

plt.xlabel("Time(weeks)", fontsize=28, fontweight='bold')
plt.xticks(fontsize=28, fontweight='bold')

plt.ylabel('Error Rate Percentage', fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
# plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x}%"))

plt.grid(True)
plt.show()

print(error_rate_percent)