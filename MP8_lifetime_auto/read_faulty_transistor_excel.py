import re


def get_faulty_transistor_equation_data(filename):

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

for data in get_faulty_transistor_equation_data(filename):
    print(data)
