from itertools import combinations
import re

def parse_log_line(line):
    pattern = r"\[\w+ \w+ \d+ \d+:\d+:\d+ \d+\] >> \[(.*?)\], \[(.*?)\], \[compliment: (True|False)\]"
    match = re.search(pattern, line)
    
    if match:
        A = list(map(int, match.group(1).split(", ")))
        B = list(map(int, match.group(2).split(", ")))
        result = True if match.group(3) == "True" else False
        return A, B, result
    return None

def load_log_file(filepath):
    input_data = []
    
    with open(filepath, 'r') as file:
        for line in file:
            parsed_data = parse_log_line(line)
            if parsed_data:
                input_data.append(parsed_data)
    
    return input_data


log_filepath = 'pattern.txt'
# log_filepath = 'MP8_lifetime_auto/pattern.txt'
input_data = load_log_file(log_filepath)




import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

X = []
y = []

for A, B, result in input_data:
    # Combine A and B into a single 16-bit input
    X.append(A + B)
    y.append(1 if result else 0)

X = np.array(X)
y = np.array(y)



poly = PolynomialFeatures(degree=1, interaction_only=False)
X_train = poly.fit_transform(X)
# X_train = X.copy()
X_test = X_train.copy()
y_train = y.copy()
y_test = y_train.copy()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("Model coefficients:", model.coef_)
