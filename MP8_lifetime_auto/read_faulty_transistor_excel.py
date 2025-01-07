from typing import List, Dict
import re


def get_faulty_transistor_equation_data(filename) -> List[Dict]:

    data_list = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        
        transistor_pattern = r"transistor: \((\d+), (\d+), (\d+)\)"
        lifetime_pattern = r"(unoptimized|dataset|normal EQ) lifetime: \[(\d+) weeks\]"
        equation_pattern = r"(\d+) >>> equation:.*\[(\d+) weeks\]"
        
        entry = {
            "transistor": None,
            "unoptimized_lifetime": None,
            "dataset_lifetime": None,
            "normal_eq_lifetime": None,
            "1": None,
            "2": None,
            "3": None,
            "4": None,
            "5": None
        }

        for line in lines:
            # Search for transistor data
            transistor_match = re.search(transistor_pattern, line)
            if transistor_match:
                # Store the transistor values as a tuple
                entry["transistor"] = tuple(map(int, transistor_match.groups()))
            
            # Search for lifetime data (unoptimized, dataset, or normal EQ)
            lifetime_match = re.search(lifetime_pattern, line)
            if lifetime_match:
                lifetime_type = lifetime_match.group(1)
                lifetime_value = int(lifetime_match.group(2))
                # Assign the lifetime value to the corresponding field in the dictionary
                if lifetime_type == "unoptimized":
                    entry["unoptimized_lifetime"] = lifetime_value
                elif lifetime_type == "dataset":
                    entry["dataset_lifetime"] = lifetime_value
                elif lifetime_type == "normal EQ":
                    entry["normal_eq_lifetime"] = lifetime_value
            
            # Search for equation lifetime data
            equation_match = re.search(equation_pattern, line)
            if equation_match:
                eq_number = equation_match.group(1)
                eq_lifetime = int(equation_match.group(2))
                # Store the equation number and corresponding lifetime
                if eq_number == "1":
                    entry["1"] = eq_lifetime
                elif eq_number == "2":
                    entry["2"] = eq_lifetime
                elif eq_number == "3":
                    entry["3"] = eq_lifetime
                elif eq_number == "4":
                    entry["4"] = eq_lifetime
                elif eq_number == "5":
                    entry["5"] = eq_lifetime

            if (line == "\n"):
                data_list.append(entry)
                # Reset the entry for the next set of data
                entry = {
                    "transistor": None,
                    "unoptimized_lifetime": None,
                    "dataset_lifetime": None,
                    "normal_eq_lifetime": None,
                    "1": None,
                    "2": None,
                    "3": None,
                    "4": None,
                    "5": None
                }

        return data_list
    

#######################################################################################

filename = "probability_detector.py.faulty_transistor_equation.log"
transistors = get_faulty_transistor_equation_data(filename)

if False:
    for t in transistors:
        print(t)





# anylysis
_tmp = []
eq_count = 0
for t in transistors:
    transistor = t["transistor"]
    unoptimized_lifetime = t["unoptimized_lifetime"]
    normal_eq_lifetimme = t["normal_eq_lifetime"]
    # dataset_lifetime = t["dataset_lifetime"]

    lifetime_1bit = t["1"] or -1
    lifetime_2bit = t["2"] or -1
    lifetime_3bit = t["3"] or -1
    lifetime_4bit = t["4"] or -1
    lifetime_5bit = t["5"] or -1
    max_equation_lifetime = max(
        lifetime_1bit, lifetime_2bit, lifetime_3bit, lifetime_4bit, lifetime_5bit
    )
    
    # normal (healthy) optimizer is doing best
    if False and t["normal_eq_lifetime"] >= t["dataset_lifetime"]:
        print(f"{t['transistor']} \tnormal EQ is best \t[{t['normal_eq_lifetime']}>{t['dataset_lifetime']},{t['unoptimized_lifetime']}]")
        eq_count += 1


    # normal equation is doing better than equation and is preffered
    if False and (normal_eq_lifetimme >= max_equation_lifetime) and (normal_eq_lifetimme > unoptimized_lifetime):
        print(f"{transistor} \tnormal EQ is better [{normal_eq_lifetimme} > {max_equation_lifetime, unoptimized_lifetime}]")
        eq_count += 1

    # no optimization can not be done on these transistors
    if False and max(max_equation_lifetime, normal_eq_lifetimme) <= unoptimized_lifetime:
        print(f"{transistor} \tunimproved unoptimized lifetime \t[{max(max_equation_lifetime, normal_eq_lifetimme)} !> {unoptimized_lifetime}]")
        eq_count += 1
        
    # normal (healthy) optimizer is not enough, and we should use a special optimizer mostly
    if True and (max_equation_lifetime > normal_eq_lifetimme) and (max_equation_lifetime > unoptimized_lifetime):
        eq_diff_unoptimized = (max_equation_lifetime - unoptimized_lifetime) / unoptimized_lifetime * 100
        eq_diff_normal = (max_equation_lifetime - normal_eq_lifetimme) / normal_eq_lifetimme * 100
        print(f"{transistor} \tEQuation is needed \t[{max_equation_lifetime} > {normal_eq_lifetimme},{unoptimized_lifetime}] \t({eq_diff_normal:2.0f}, {eq_diff_unoptimized:2.0f})% \t{"!!" if eq_diff_normal<11 else ""} \t{"!!" if eq_diff_unoptimized<11 else ""}")
        
        _tmp.append(t)
        eq_count += 1

print(f"count: {eq_count}")

print("+"*20)
_tmp = sorted(_tmp, reverse=True, key=lambda x: (x["dataset_lifetime"] - x["unoptimized_lifetime"])/x["unoptimized_lifetime"])
for t in _tmp:
    dataset_diff_normal = (t['dataset_lifetime'] - t['normal_eq_lifetime']) / t['normal_eq_lifetime'] * 100

    print(f"{t['transistor']} \tdataset:{t['dataset_lifetime']:3.0f} <- {t['unoptimized_lifetime']} \tnormal:{t['normal_eq_lifetime']} ")
    # if not dataset_diff_normal < 16:
    #     print(f"{t['transistor']} \t{t['dataset_lifetime']:3.0f} <- {t['unoptimized_lifetime']} \tnormal:{t['normal_eq_lifetime']} ")
    # else:
    #     print(f"!!!{t['transistor']} \t{t['dataset_lifetime']:3.0f} <- {t['unoptimized_lifetime']} \tnormal:{t['normal_eq_lifetime']} ")


# analysis output
"""
critical failure transistor, no improvement is possible

(0, 0, 2)       unimproved unoptimized lifetime         [68 !> 68]
(0, 0, 5)       unimproved unoptimized lifetime         [68 !> 68]
(1, 0, 2)       unimproved unoptimized lifetime         [88 !> 88]
(1, 0, 5)       unimproved unoptimized lifetime         [88 !> 88]
(5, 5, 1)       unimproved unoptimized lifetime         [95 !> 98]
(5, 6, 1)       unimproved unoptimized lifetime         [82 !> 98]
(5, 7, 0)       unimproved unoptimized lifetime         [85 !> 98]
(6, 5, 1)       unimproved unoptimized lifetime         [88 !> 98]
(6, 6, 1)       unimproved unoptimized lifetime         [82 !> 98]
(6, 7, 0)       unimproved unoptimized lifetime         [82 !> 98]
count: 10
total: 337
"""


"""
normal equation lifetime is best, it is better or equal to max_equation_lifetime

(0, 0, 1)       normal EQ is better [169 > (115, 98)]
(0, 0, 1)       normal EQ is better [169 > (169, 98)]
(0, 0, 3)       normal EQ is better [169 > (169, 98)]
(0, 0, 4)       normal EQ is better [169 > (169, 98)]
(0, 1, 1)       normal EQ is better [169 > (169, 98)]
(0, 1, 3)       normal EQ is better [169 > (169, 98)]
(0, 1, 4)       normal EQ is better [169 > (169, 98)]
(0, 2, 1)       normal EQ is better [169 > (169, 98)]
(0, 2, 3)       normal EQ is better [169 > (169, 98)]
(0, 2, 4)       normal EQ is better [169 > (169, 98)]
(0, 3, 1)       normal EQ is better [169 > (169, 98)]
(0, 3, 3)       normal EQ is better [169 > (169, 98)]
(0, 3, 4)       normal EQ is better [169 > (169, 98)]
(0, 4, 1)       normal EQ is better [169 > (169, 98)]
(0, 4, 3)       normal EQ is better [169 > (169, 98)]
(0, 4, 4)       normal EQ is better [169 > (169, 98)]
(0, 5, 0)       normal EQ is better [87 > (87, 38)]
(0, 5, 1)       normal EQ is better [146 > (146, 98)]
(0, 5, 3)       normal EQ is better [129 > (129, 98)]
(0, 5, 4)       normal EQ is better [129 > (129, 98)]
(0, 6, 0)       normal EQ is better [169 > (169, 98)]
(0, 6, 2)       normal EQ is better [169 > (169, 98)]
(0, 6, 5)       normal EQ is better [169 > (169, 98)]
(0, 7, 0)       normal EQ is better [169 > (169, 98)]
(0, 7, 1)       normal EQ is better [113 > (113, 98)]
(0, 7, 3)       normal EQ is better [169 > (169, 98)]
(0, 7, 4)       normal EQ is better [169 > (169, 98)]
(1, 0, 1)       normal EQ is better [169 > (169, 98)]
(1, 0, 3)       normal EQ is better [145 > (145, 98)]
(1, 0, 4)       normal EQ is better [145 > (145, 98)]
(1, 1, 1)       normal EQ is better [169 > (169, 98)]
(1, 1, 3)       normal EQ is better [145 > (145, 98)]
(1, 1, 4)       normal EQ is better [145 > (145, 98)]
(1, 2, 1)       normal EQ is better [169 > (169, 98)]
(1, 3, 1)       normal EQ is better [169 > (169, 98)]
(1, 3, 3)       normal EQ is better [145 > (145, 98)]
(1, 3, 4)       normal EQ is better [145 > (145, 98)]
(1, 5, 0)       normal EQ is better [169 > (169, 98)]
(1, 5, 2)       normal EQ is better [145 > (145, 98)]
(1, 5, 5)       normal EQ is better [145 > (145, 98)]
(1, 6, 1)       normal EQ is better [169 > (169, 98)]
(1, 6, 3)       normal EQ is better [113 > (98, 98)]
(1, 6, 4)       normal EQ is better [113 > (98, 98)]
(1, 7, 0)       normal EQ is better [169 > (169, 98)]
(1, 7, 1)       normal EQ is better [34 > (34, 20)]
(1, 7, 3)       normal EQ is better [169 > (169, 98)]
(1, 7, 4)       normal EQ is better [169 > (169, 98)]
(2, 0, 1)       normal EQ is better [145 > (145, 98)]
(2, 0, 2)       normal EQ is better [99 > (99, 98)]
(2, 0, 3)       normal EQ is better [127 > (127, 98)]
(2, 0, 4)       normal EQ is better [127 > (127, 98)]
(2, 0, 5)       normal EQ is better [99 > (99, 98)]
(2, 1, 1)       normal EQ is better [145 > (145, 98)]
(2, 1, 2)       normal EQ is better [99 > (99, 98)]
(2, 1, 3)       normal EQ is better [127 > (127, 98)]
(2, 1, 4)       normal EQ is better [127 > (127, 98)]
(2, 1, 5)       normal EQ is better [99 > (99, 98)]
(2, 2, 1)       normal EQ is better [145 > (84, 98)]
(2, 3, 2)       normal EQ is better [114 > (114, 98)]
(2, 3, 3)       normal EQ is better [111 > (106, 98)]
(2, 3, 4)       normal EQ is better [111 > (106, 98)]
(2, 3, 5)       normal EQ is better [114 > (114, 98)]
(2, 4, 0)       normal EQ is better [154 > (154, 98)]
(2, 6, 1)       normal EQ is better [153 > (153, 98)]
(2, 6, 3)       normal EQ is better [117 > (83, 98)]
(2, 6, 4)       normal EQ is better [117 > (83, 98)]
(2, 7, 0)       normal EQ is better [169 > (169, 98)]
(2, 7, 3)       normal EQ is better [169 > (169, 98)]
(2, 7, 4)       normal EQ is better [169 > (169, 98)]
(3, 0, 2)       normal EQ is better [106 > (106, 98)]
(3, 0, 3)       normal EQ is better [120 > (120, 98)]
(3, 0, 4)       normal EQ is better [120 > (120, 98)]
(3, 0, 5)       normal EQ is better [106 > (106, 98)]
(3, 1, 1)       normal EQ is better [127 > (127, 98)]
(3, 1, 2)       normal EQ is better [106 > (106, 98)]
(3, 1, 3)       normal EQ is better [120 > (120, 98)]
(3, 1, 4)       normal EQ is better [120 > (120, 98)]
(3, 1, 5)       normal EQ is better [106 > (106, 98)]
(3, 2, 1)       normal EQ is better [107 > (83, 98)]
(3, 2, 3)       normal EQ is better [112 > (106, 98)]
(3, 2, 4)       normal EQ is better [112 > (106, 98)]
(3, 4, 3)       normal EQ is better [109 > (94, 98)]
(3, 4, 4)       normal EQ is better [109 > (94, 98)]
(3, 5, 1)       normal EQ is better [126 > (94, 98)]
(3, 5, 3)       normal EQ is better [116 > (94, 98)]
(3, 5, 4)       normal EQ is better [116 > (94, 98)]
(3, 6, 1)       normal EQ is better [114 > (107, 98)]
(3, 6, 3)       normal EQ is better [117 > (95, 98)]
(3, 6, 4)       normal EQ is better [117 > (95, 98)]
(3, 7, 0)       normal EQ is better [132 > (98, 98)]
(3, 7, 1)       normal EQ is better [96 > (65, 65)]
(3, 7, 3)       normal EQ is better [169 > (169, 98)]
(3, 7, 4)       normal EQ is better [169 > (169, 98)]
(4, 0, 2)       normal EQ is better [109 > (109, 98)]
(4, 0, 3)       normal EQ is better [116 > (116, 98)]
(4, 0, 4)       normal EQ is better [116 > (116, 98)]
(4, 0, 5)       normal EQ is better [109 > (109, 98)]
(4, 1, 1)       normal EQ is better [108 > (83, 98)]
(4, 1, 2)       normal EQ is better [113 > (109, 98)]
(4, 1, 5)       normal EQ is better [113 > (109, 98)]
(4, 3, 1)       normal EQ is better [118 > (110, 98)]
(4, 3, 3)       normal EQ is better [110 > (83, 98)]
(4, 3, 4)       normal EQ is better [110 > (83, 98)]
(4, 4, 1)       normal EQ is better [117 > (107, 98)]
(4, 4, 2)       normal EQ is better [114 > (112, 98)]
(4, 4, 3)       normal EQ is better [111 > (83, 98)]
(4, 4, 4)       normal EQ is better [111 > (83, 98)]
(4, 4, 5)       normal EQ is better [114 > (112, 98)]
(4, 5, 1)       normal EQ is better [106 > (83, 98)]
(4, 5, 3)       normal EQ is better [112 > (106, 98)]
(4, 5, 4)       normal EQ is better [112 > (106, 98)]
(4, 6, 0)       normal EQ is better [135 > (84, 89)]
(4, 6, 3)       normal EQ is better [118 > (95, 98)]
(4, 6, 4)       normal EQ is better [118 > (95, 98)]
(4, 7, 0)       normal EQ is better [101 > (82, 98)]
(4, 7, 1)       normal EQ is better [125 > (83, 83)]
(4, 7, 3)       normal EQ is better [169 > (169, 98)]
(4, 7, 4)       normal EQ is better [169 > (169, 98)]
(5, 0, 2)       normal EQ is better [111 > (111, 98)]
(5, 0, 3)       normal EQ is better [114 > (114, 98)]
(5, 0, 4)       normal EQ is better [114 > (114, 98)]
(5, 0, 5)       normal EQ is better [111 > (111, 98)]
(5, 1, 2)       normal EQ is better [115 > (115, 98)]
(5, 1, 5)       normal EQ is better [115 > (115, 98)]
(5, 2, 1)       normal EQ is better [115 > (111, 98)]
(5, 2, 3)       normal EQ is better [111 > (106, 98)]
(5, 2, 4)       normal EQ is better [111 > (106, 98)]
(5, 3, 1)       normal EQ is better [114 > (83, 98)]
(5, 3, 2)       normal EQ is better [113 > (112, 98)]
(5, 3, 3)       normal EQ is better [112 > (83, 98)]
(5, 3, 4)       normal EQ is better [112 > (83, 98)]
(5, 3, 5)       normal EQ is better [113 > (112, 98)]
(5, 4, 0)       normal EQ is better [119 > (108, 98)]
(5, 4, 1)       normal EQ is better [106 > (83, 98)]
(5, 4, 2)       normal EQ is better [114 > (114, 98)]
(5, 4, 3)       normal EQ is better [111 > (83, 98)]
(5, 4, 4)       normal EQ is better [111 > (83, 98)]
(5, 4, 5)       normal EQ is better [114 > (114, 98)]
(5, 5, 3)       normal EQ is better [112 > (83, 98)]
(5, 5, 4)       normal EQ is better [112 > (83, 98)]
(5, 6, 0)       normal EQ is better [154 > (70, 98)]
(5, 6, 3)       normal EQ is better [119 > (107, 98)]
(5, 6, 4)       normal EQ is better [119 > (107, 98)]
(5, 7, 1)       normal EQ is better [149 > (95, 95)]
(5, 7, 3)       normal EQ is better [169 > (169, 98)]
(5, 7, 4)       normal EQ is better [169 > (169, 98)]
(6, 0, 2)       normal EQ is better [112 > (112, 98)]
(6, 0, 3)       normal EQ is better [113 > (113, 98)]
(6, 0, 4)       normal EQ is better [113 > (113, 98)]
(6, 0, 5)       normal EQ is better [112 > (112, 98)]
(6, 1, 1)       normal EQ is better [113 > (112, 98)]
(6, 1, 2)       normal EQ is better [112 > (108, 98)]
(6, 1, 5)       normal EQ is better [112 > (108, 98)]
(6, 2, 1)       normal EQ is better [113 > (105, 98)]
(6, 3, 0)       normal EQ is better [118 > (85, 98)]
(6, 3, 1)       normal EQ is better [108 > (83, 98)]
(6, 3, 2)       normal EQ is better [111 > (83, 98)]
(6, 3, 3)       normal EQ is better [114 > (112, 98)]
(6, 3, 4)       normal EQ is better [114 > (112, 98)]
(6, 3, 5)       normal EQ is better [111 > (83, 98)]
(6, 4, 0)       normal EQ is better [127 > (117, 98)]
(6, 4, 1)       normal EQ is better [100 > (83, 98)]
(6, 4, 2)       normal EQ is better [111 > (83, 98)]
(6, 4, 3)       normal EQ is better [114 > (112, 98)]
(6, 4, 4)       normal EQ is better [114 > (112, 98)]
(6, 4, 5)       normal EQ is better [111 > (83, 98)]
(6, 5, 0)       normal EQ is better [143 > (117, 98)]
(6, 5, 2)       normal EQ is better [112 > (83, 98)]
(6, 5, 5)       normal EQ is better [112 > (83, 98)]
(6, 6, 0)       normal EQ is better [167 > (31, 98)]
(6, 6, 2)       normal EQ is better [119 > (107, 98)]
(6, 6, 5)       normal EQ is better [119 > (107, 98)]
(6, 7, 1)       normal EQ is better [165 > (165, 98)]
(6, 7, 2)       normal EQ is better [169 > (169, 98)]
count: 174
"""


"""
transistors that need special optimizer, data is sorted with percentage of lifetime increase in the top

(2, 7, 1)       dataset:137 <- 43       normal:63 
(0, 6, 1)       dataset:112 <- 38       normal:51 
(0, 7, 2)       dataset:112 <- 38       normal:68 
(0, 7, 5)       dataset:112 <- 38       normal:68 
(1, 7, 2)       dataset:112 <- 38       normal:51 
(1, 7, 5)       dataset:112 <- 38       normal:51 
(2, 7, 2)       dataset:112 <- 38       normal:51 
(2, 7, 5)       dataset:112 <- 38       normal:51 
(3, 7, 2)       dataset:112 <- 38       normal:51 
(3, 7, 5)       dataset:112 <- 38       normal:51 
(4, 7, 2)       dataset:112 <- 38       normal:51 
(4, 7, 5)       dataset:112 <- 38       normal:51 
(5, 7, 2)       dataset:112 <- 38       normal:51 
(5, 7, 5)       dataset:112 <- 38       normal:51 
(6, 7, 3)       dataset:111 <- 38       normal:51 
(6, 7, 4)       dataset:111 <- 38       normal:51 
(1, 6, 2)       dataset:190 <- 74       normal:112 
(1, 6, 5)       dataset:190 <- 74       normal:112 
(1, 5, 1)       dataset:169 <- 68       normal:62 
(2, 6, 2)       dataset:189 <- 79       normal:108 
(2, 6, 5)       dataset:189 <- 79       normal:108 
(1, 3, 0)       dataset:162 <- 68       normal:68 
(2, 6, 0)       dataset:146 <- 62       normal:83 
(1, 4, 0)       dataset:160 <- 68       normal:112 
(0, 3, 0)       dataset: 89 <- 38       normal:38 
(0, 4, 0)       dataset: 89 <- 38       normal:38 
(3, 6, 2)       dataset:189 <- 81       normal:108 
(3, 6, 5)       dataset:189 <- 81       normal:108 
(1, 2, 0)       dataset:158 <- 68       normal:68 
(0, 1, 0)       dataset: 88 <- 38       normal:38 
(4, 6, 2)       dataset:189 <- 82       normal:107 
(4, 6, 5)       dataset:189 <- 82       normal:107 
(5, 6, 2)       dataset:189 <- 82       normal:107 
(5, 6, 5)       dataset:189 <- 82       normal:107 
(6, 6, 3)       dataset:189 <- 82       normal:106 
(6, 6, 4)       dataset:189 <- 82       normal:106 
(0, 2, 0)       dataset: 85 <- 38       normal:38 
(2, 5, 0)       dataset:162 <- 73       normal:86 
(3, 5, 0)       dataset:175 <- 83       normal:100 
(0, 6, 3)       dataset:143 <- 68       normal:68 
(0, 6, 4)       dataset:143 <- 68       normal:68 
(3, 6, 0)       dataset:161 <- 77       normal:111 
(2, 5, 2)       dataset:187 <- 90       normal:112 
(2, 5, 5)       dataset:187 <- 90       normal:112 
(0, 5, 2)       dataset:141 <- 68       normal:98 
(0, 5, 5)       dataset:141 <- 68       normal:98 
(2, 3, 0)       dataset:181 <- 88       normal:119 
(1, 6, 0)       dataset: 96 <- 47       normal:62 
(0, 4, 2)       dataset:138 <- 68       normal:68 
(0, 4, 5)       dataset:138 <- 68       normal:68 
(3, 5, 2)       dataset:188 <- 93       normal:109 
(3, 5, 5)       dataset:188 <- 93       normal:109 
(1, 1, 0)       dataset:137 <- 68       normal:68 
(5, 5, 2)       dataset:189 <- 95       normal:113 
(5, 5, 5)       dataset:189 <- 95       normal:113 
(6, 5, 3)       dataset:189 <- 95       normal:113 
(6, 5, 4)       dataset:189 <- 95       normal:113 
(0, 3, 2)       dataset:135 <- 68       normal:68 
(0, 3, 5)       dataset:135 <- 68       normal:68 
(4, 5, 2)       dataset:188 <- 95       normal:113 
(4, 5, 5)       dataset:188 <- 95       normal:113 
(1, 4, 2)       dataset:174 <- 88       normal:111 
(1, 4, 5)       dataset:174 <- 88       normal:111 
(2, 4, 1)       dataset:172 <- 88       normal:82 
(4, 5, 0)       dataset:179 <- 93       normal:120 
(1, 5, 3)       dataset:169 <- 88       normal:88 
(1, 5, 4)       dataset:169 <- 88       normal:88 
(3, 4, 0)       dataset:171 <- 90       normal:100 
(5, 5, 0)       dataset:184 <- 98       normal:134 
(2, 2, 0)       dataset:163 <- 88       normal:88 
(1, 0, 0)       dataset:125 <- 68       normal:68 
(4, 4, 0)       dataset:176 <- 96       normal:108 
(2, 1, 0)       dataset:161 <- 88       normal:88 
(1, 3, 2)       dataset:160 <- 88       normal:88 
(1, 3, 5)       dataset:160 <- 88       normal:88 
(2, 0, 0)       dataset:160 <- 88       normal:88 
(3, 2, 0)       dataset:178 <- 98       normal:118 
(3, 4, 2)       dataset:178 <- 98       normal:116 
(3, 4, 5)       dataset:178 <- 98       normal:116 
(0, 2, 2)       dataset:123 <- 68       normal:68 
(0, 2, 5)       dataset:123 <- 68       normal:68 
(4, 1, 0)       dataset:176 <- 98       normal:117 
(5, 0, 0)       dataset:176 <- 98       normal:116 
(1, 2, 2)       dataset:158 <- 88       normal:88 
(1, 2, 5)       dataset:158 <- 88       normal:88 
(0, 0, 0)       dataset: 68 <- 38       normal:38 
(0, 0, 0)       dataset: 68 <- 38       normal:38 
(3, 2, 2)       dataset:172 <- 98       normal:113 
(3, 2, 5)       dataset:172 <- 98       normal:113 
(5, 3, 0)       dataset:172 <- 98       normal:111 
(2, 4, 3)       dataset:171 <- 98       normal:100 
(2, 4, 4)       dataset:171 <- 98       normal:100 
(3, 3, 1)       dataset:168 <- 98       normal:95 
(4, 3, 0)       dataset:168 <- 98       normal:107 
(4, 3, 2)       dataset:167 <- 98       normal:115 
(4, 3, 5)       dataset:167 <- 98       normal:115 
(6, 2, 0)       dataset:166 <- 98       normal:112 
(3, 3, 3)       dataset:165 <- 98       normal:105 
(3, 3, 4)       dataset:165 <- 98       normal:105 
(4, 2, 1)       dataset:164 <- 98       normal:103 
(5, 2, 0)       dataset:164 <- 98       normal:110 
(5, 2, 2)       dataset:163 <- 98       normal:114 
(5, 2, 5)       dataset:163 <- 98       normal:114 
(6, 1, 0)       dataset:163 <- 98       normal:112 
(6, 2, 3)       dataset:163 <- 98       normal:113 
(6, 2, 4)       dataset:163 <- 98       normal:113 
(3, 1, 0)       dataset:162 <- 98       normal:99 
(5, 1, 1)       dataset:162 <- 98       normal:107 
(4, 2, 3)       dataset:161 <- 98       normal:110 
(4, 2, 4)       dataset:161 <- 98       normal:110 
(6, 0, 1)       dataset:161 <- 98       normal:109 
(3, 0, 0)       dataset:160 <- 98       normal:99 
(4, 0, 0)       dataset:160 <- 98       normal:106 
(2, 2, 2)       dataset:159 <- 98       normal:99 
(2, 2, 5)       dataset:159 <- 98       normal:99 
(5, 1, 0)       dataset:158 <- 98       normal:118 
(6, 0, 0)       dataset:158 <- 98       normal:116 
(6, 1, 3)       dataset:158 <- 98       normal:113 
(6, 1, 4)       dataset:158 <- 98       normal:113 
(4, 0, 1)       dataset:156 <- 98       normal:120 
(4, 2, 0)       dataset:156 <- 98       normal:123 
(5, 1, 3)       dataset:156 <- 98       normal:110 
(5, 1, 4)       dataset:156 <- 98       normal:110 
(3, 3, 0)       dataset:154 <- 98       normal:133 
(4, 2, 2)       dataset:153 <- 98       normal:116 
(4, 2, 5)       dataset:153 <- 98       normal:116 
(3, 3, 2)       dataset:152 <- 98       normal:120 
(3, 3, 5)       dataset:152 <- 98       normal:120 
(6, 2, 2)       dataset:152 <- 98       normal:112 
(6, 2, 5)       dataset:152 <- 98       normal:112 
(3, 4, 1)       dataset:151 <- 98       normal:126 
(2, 5, 1)       dataset:150 <- 98       normal:147 
(1, 1, 2)       dataset:134 <- 88       normal:88 
(1, 1, 5)       dataset:134 <- 88       normal:88 
(2, 4, 2)       dataset:148 <- 98       normal:127 
(2, 4, 5)       dataset:148 <- 98       normal:127 
(4, 1, 3)       dataset:147 <- 98       normal:112 
(4, 1, 4)       dataset:147 <- 98       normal:112 
(0, 1, 2)       dataset:101 <- 68       normal:68 
(0, 1, 5)       dataset:101 <- 68       normal:68 
(5, 0, 1)       dataset:144 <- 98       normal:109 
(2, 2, 3)       dataset:143 <- 98       normal:127 
(2, 2, 4)       dataset:143 <- 98       normal:127 
(1, 2, 3)       dataset:134 <- 98       normal:145 
(1, 2, 4)       dataset:134 <- 98       normal:145 
(2, 5, 3)       dataset:134 <- 98       normal:113 
(2, 5, 4)       dataset:134 <- 98       normal:113 
(3, 0, 1)       dataset:134 <- 98       normal:127 
(1, 4, 3)       dataset:114 <- 98       normal:114 
(1, 4, 4)       dataset:114 <- 98       normal:114 
(1, 4, 1)       dataset:113 <- 98       normal:113 
(2, 3, 1)       dataset:107 <- 98       normal:107 
(4, 6, 1)       dataset:104 <- 98       normal:94 
"""