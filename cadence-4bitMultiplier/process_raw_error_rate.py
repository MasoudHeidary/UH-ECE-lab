import re
import matplotlib.pyplot as plt 

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


for t_week in [20, 40, 60, 80, 100]:

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
            
            


    # _normal = max(filter_data(normal_file.read()))
    # _modified = max(filter_data(modified_file.read()))
    # _m2 = max(filter_data(M2_file.read()))
    # _m3 = max(filter_data(M3_file.read()))
    # normal_delay += [_normal]
    # modified_delay += [_modified]
    # M2_delay += [_m2]
    # M3_delay += [_m3]

    # print(f"{_modified}, ")
    



# plt.plot(x, normal_delay, label="Normal Multiplier", linewidth=2.5)
# plt.plot(x, modified_delay, label="Modified", linewidth=2.5)
# plt.plot(x, M2_delay, label="M2", linewidth=2.5)
# plt.plot(x, M3_delay, label="M3", linewidth=2.5)
# plt.xlabel('time(weeks)', fontsize=16)
# plt.ylabel('delay(ps)', fontsize=16)
# plt.title('Delay', fontsize=16)
# plt.tick_params(axis='both', labelsize=16)
# plt.legend()
# plt.grid(True)
# plt.show()
