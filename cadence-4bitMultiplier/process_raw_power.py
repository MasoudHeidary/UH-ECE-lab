import re
import matplotlib.pyplot as plt

from output_delay import normal_delay, modified_delay

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


x = []
normal_power = []
modified_power = []

for i in range(1, 100, 1):
    x += [i]

    normal_power_file = open(f"./raw_data_power/power_N-{i}.txt", "r")
    modified_power_file = open(f"./raw_data_power/power_M-{i}.txt", "r")

    _normal = extract_number(normal_power_file.read())
    _modified = extract_number(modified_power_file.read())
    normal_power += [_normal]
    modified_power += [_modified]


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

normal_energy = [i/1000 for i in normal_energy]
modified_energy = [i/1000 for i in modified_energy]

# plt.title('Energy', fontsize=16)
plt.plot(x, normal_energy, label="normal", linewidth=3)
plt.plot(x, modified_energy, label="modified", linewidth=3)

plt.xlabel('time(weeks)', fontsize=14, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')

plt.ylabel('energy(fJ)', fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

# plt.tick_params(axis='both', labelsize=14)

plt.legend(fontsize=14)
plt.grid(True)
plt.show()

# from time import sleep
# sleep(1000)