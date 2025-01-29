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

    # print(f"{_m3}, ")
    

# expand time
x = [i*2 for i in x]

# check the degredation percentage
# d = normal_delay[-1] - normal_delay[0]
# print((modified_delay[-1]-modified_delay[0]-d)/d * 100)
# print((M3_delay[-1]-modified_delay[0]-d)/d * 100)
# print((M2_delay[-1]-modified_delay[0]-d)/d * 100)
# exit()

# normalize delay plot
normalize_scale = normal_delay[0]
def normalize(x):
    return x/normalize_scale
modified_delay = list(map(normalize, modified_delay))
M3_delay = list(map(normalize, M3_delay))
M2_delay = list(map(normalize, M2_delay))
normal_delay = list(map(normalize, normal_delay))



################################################### plot
if False:
    plt.figure(figsize=(13, 10))

    plt.plot(x, modified_delay, label="100% tampered", linewidth=5)
    plt.plot(x, M3_delay, label="50% tampered", linewidth=5)
    plt.plot(x, M2_delay, label="33% tampered", linewidth=5)
    plt.plot(x, normal_delay, label="0%  tampered", linewidth=5)

    plt.xlabel('Time(weeks)', fontsize=28, fontweight='bold')
    plt.xticks(fontsize=28, fontweight='bold')

    plt.ylabel('Normalized Execution Time', fontsize=28, fontweight='bold')
    plt.yticks(fontsize=28, fontweight='bold')

    # plt.title('Delay', fontsize=16)
    # plt.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.show()

if True:
    plt.figure(figsize=(13, 10))

    plt.plot(x, modified_delay, label="Attacked", linewidth=8)
    plt.plot(x, normal_delay, label="Normal", linewidth=8)

    plt.xlabel('Time(weeks)', fontsize=36, fontweight='bold')
    plt.xticks(fontsize=36, fontweight='bold')

    plt.ylabel('Normalized Execution Time', fontsize=36, fontweight='bold')
    plt.yticks(fontsize=36, fontweight='bold')

    # plt.title('Delay', fontsize=16)
    # plt.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=36)
    plt.grid(True)
    plt.show()


# compare aging to normal aging
# a = M2_delay[-1] - normal_delay[0]
# w = normal_delay[-1] - normal_delay[0]
# print((a-w ) / w * 100)

# a = M3_delay[-1] - normal_delay[0]
# print((a-w ) / w * 100)

# a = modified_delay[-1] - normal_delay[0]
# print((a-w ) / w * 100)

a = modified_delay[25-1]  
b = normal_delay[0]
print( (a-b ) / b * 100)
#-------------

print(
    (M2_delay[-1] - normal_delay[0]) / normal_delay[0] * 100
)


b = normal_delay[-1]
a = normal_delay[0]
print((b-a)/a * 100)


w = normal_delay[0]
a = modified_delay[24]
print((a-w)/w * 100)