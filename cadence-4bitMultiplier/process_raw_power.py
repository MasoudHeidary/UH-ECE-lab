import re
import matplotlib.pyplot as plt

from output_delay import *

def extract_number(text):
    # Regular expression to match numbers (including decimals) followed by the unit 'u'
    pattern = r'(\d+(\.\d+)?)(?=u)'
    
    # Find the first match of the pattern in the text
    match = re.search(pattern, text)
    
    if match:
        # Extract the matched number
        number = float(match.group(1))
        return number
    else:
        return None


t = []
normal_power = []
modified_power = []
M33_power = []
M50_power = []

for i in range(1, 100, 1):
    t += [i]

    normal_power_file = open(f"./raw_data_power/power_N-{i}.txt", "r")
    modified_power_file = open(f"./raw_data_power/power_M-{i}.txt", "r")
    M33_power_file = open(f"./raw_data_power/power33%-{i}.txt", "r")
    M50_power_file = open(f"./raw_data_power/power50%-{i}.txt", "r")

    _normal = extract_number(normal_power_file.read())
    _modified = extract_number(modified_power_file.read())
    _33 = extract_number(M33_power_file.read())
    _50 = extract_number(M50_power_file.read())

    normal_power += [_normal]
    modified_power += [_modified]
    M33_power += [_33]
    M50_power += [_50]


# expand time
t = [i*2 for i in t]

if False:
    plt.title('Power', fontsize=16)
    plt.plot(x, normal_power, label="normal", linewidth=2.5)
    plt.plot(x, modified_power, label="modified", linewidth=2.5)
    plt.xlabel('time(weeks)', fontsize=16)
    plt.ylabel('power(uW)', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()


normal_energy = [i*j for (i,j) in zip(normal_power, normal_delay)]
modified_energy = [i*j for (i,j) in zip (modified_power, modified_delay)]
M33_energy = [i*j for (i,j) in zip(M33_power, M33_delay)]
M50_energy = [i*j for (i,j) in zip(M50_power, M50_delay)]

normal_energy = [i/1000 for i in normal_energy]
modified_energy = [i/1000 for i in modified_energy]
M33_energy = [i/1000 for i in M33_energy]
M50_energy = [i/1000 for i in M50_energy]


# compare growth
# d = normal_energy[24] - normal_energy[0]
# print((modified_energy[24]-normal_energy[0]-d)/d *100)
# d = normal_energy[49] - normal_energy[0]
# print((modified_energy[49]-normal_energy[0]-d)/d *100)
# d = normal_energy[74] - normal_energy[0]
# print((modified_energy[74]-normal_energy[0]-d)/d *100)
# d = normal_energy[98] - normal_energy[0]
# print((modified_energy[98]-normal_energy[0]-d)/d *100)
# exit()

# normalize energy
normalize_scale = normal_energy[0]
def normalize(x):
    return x/normalize_scale
normal_energy = list(map(normalize, normal_energy))
modified_energy = list(map(normalize, modified_energy))
M33_energy = list(map(normalize, M33_energy))
M50_energy = list(map(normalize, M50_energy))


plt.figure(figsize=(13, 10))

# plt.title('Energy', fontsize=16)
plt.plot(t, modified_energy, label="100% modified", linewidth=5)
plt.plot(t, M50_energy, label="50% modified", linewidth=5)
plt.plot(t, M33_energy, label="33% modified", linewidth=5)
plt.plot(t, normal_energy, label="0% modified", linewidth=5)

plt.xlabel('time(weeks)', fontsize=18, fontweight='bold')
plt.xticks(fontsize=18, fontweight='bold')

plt.ylabel('normalized energy', fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')

# plt.tick_params(axis='both', labelsize=14)

plt.legend(fontsize=16)
plt.grid(True)
plt.show()

# from time import sleep
# sleep(1000)