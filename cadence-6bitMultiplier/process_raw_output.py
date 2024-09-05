import re
import matplotlib.pyplot as plt 




def filter_data(data):
    pattern = r"(\d+)\s+([\d.]+)p"
    matches = re.findall(pattern, data)
    filtered_matches = [float(match[1]) for match in matches]

    return filtered_matches


t_week = []
normal_delay = []
improved_delay = []

for i in range(0, 200, 1):

    normal_file = open(f"./raw_data_FA[4][5]_T0_(2)/Normal-{i}.txt")
    improved_file = open(f"./raw_data_FA[4][5]_T0_(2)/improved-FA[4][5]-T0-{i}.txt")

    t_week += [i]
    _normal = max(filter_data(normal_file.read()))
    _improved = max(filter_data(improved_file.read()))
    normal_delay += [_normal]
    improved_delay += [_improved]





# plot
if True:
    plt.figure(figsize=(13, 10))

    plt.plot(t_week, normal_delay, label="normal", linewidth=5)
    plt.plot(t_week, improved_delay, label="improved", linewidth=5)

    plt.xlabel('Time(weeks', fontsize=28, fontweight='bold')
    plt.xticks(fontsize=28, fontweight='bold')

    plt.ylabel('Delay', fontsize=28, fontweight='bold')
    plt.yticks(fontsize=28, fontweight='bold')

    plt.legend(fontsize=28)
    plt.grid()
    plt.show()