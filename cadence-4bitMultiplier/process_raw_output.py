import re
import matplotlib.pyplot as plt 

def filter_data(data):
    pattern = r"(\d+)\s+([\d.]+)p"
    matches = re.findall(pattern, data)

    specified_values = ['7', '6', '5', '0']
    filtered_matches = [float(match[1]) for match in matches if match[0] in specified_values]

    return filtered_matches


x = []
normal_delay = []
modified_delay = []
M2_delay = []
M3_delay = []

for i in range(1, 100, 1):

    normal_file = open(f"./raw_data/N-{i}.txt")
    modified_file = open(f"./raw_data/M-{i}.txt")
    M2_file = open(f"./raw_data/M2-{i}.txt")
    M3_file = open(f"./raw_data/M3-{i}.txt")

    x += [i]
    _normal = max(filter_data(normal_file.read()))
    _modified = max(filter_data(modified_file.read()))
    _m2 = max(filter_data(M2_file.read()))
    _m3 = max(filter_data(M3_file.read()))
    normal_delay += [_normal]
    modified_delay += [_modified]
    M2_delay += [_m2]
    M3_delay += [_m3]

    # print(f"{_modified}, ")
    



plt.plot(x, normal_delay, label="Normal Multiplier", linewidth=2.5)
plt.plot(x, modified_delay, label="Modified", linewidth=2.5)
plt.plot(x, M2_delay, label="M2", linewidth=2.5)
plt.plot(x, M3_delay, label="M3", linewidth=2.5)
plt.xlabel('time(weeks)', fontsize=16)
plt.ylabel('delay(ps)', fontsize=16)
plt.title('Delay', fontsize=16)
plt.tick_params(axis='both', labelsize=16)
plt.legend()
plt.grid(True)
plt.show()
