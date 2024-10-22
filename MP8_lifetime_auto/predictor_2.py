import re
import os
import sys

from tool.log import Log
from msimulator.get_alpha_class import MultiplierStressTest

from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
import scipy.sparse as sp
from sklearn.utils import shuffle

bit_len = 8
log = Log("predictor_2.py.txt", terminal=True)

r_bin = MultiplierStressTest.reverse_signed_b

def parse_log_line(line):
    pattern = r"\[\w+ \w+ \d+ \d+:\d+:\d+ \d+\] >> \[(.*?)\], \[(.*?)\], \[compliment: (True|False)\]"
    match = re.search(pattern, line)
    
    if match:
        A = list(map(int, match.group(1).split(", ")))
        B = list(map(int, match.group(2).split(", ")))
        result = True if match.group(3) == "True" else False
        return A, B, result
    return None


def yield_load_log_files(bit_len, directory):
    # Iterate over all possible fa_i, fa_j, and t_index values
    for fa_i in range(bit_len - 1):
        for fa_j in range(bit_len):
            for t_index in range(6):
                # Generate the file name based on fa_i, fa_j, t_index
                filename = f"fa_i-{fa_i}-fa_j-{fa_j}-t_index-{t_index}.txt"
                filepath = os.path.join(directory, filename)
                log.println(f"file reading {filepath}")
                
                if not os.path.exists(filepath):
                    log.println(f"File not found: {filepath}")
                    continue
                
                # Read the file and extract data
                with open(filepath, 'r') as file:
                    for line in file:
                        parsed_data = parse_log_line(line)
                        if parsed_data:
                            A, B, result = parsed_data
                            # Yield the data instead of appending it to a list
                            yield (A, B, result, fa_i, fa_j, t_index)


def batch_load_log_files(file_path, bit_len, batch_size):
    input_data = yield_load_log_files(bit_len, file_path)
    X_batch, y_batch = [], []

    for A, B, result, fa_i, fa_j, t_index in input_data:
        X_batch.append(A + B + [fa_i, fa_j, t_index])
        y_batch.append(1 if result else 0)

        if len(X_batch) == batch_size:
            yield np.array(X_batch), np.array(y_batch)
            X_batch, y_batch = [], []

    # Yield remaining data
    if X_batch:
        yield np.array(X_batch), np.array(y_batch)


log.println("reading dataset")

batch_size = 2**16 * 4
polynomial_degree = 2
polynomial_interaction_only = False

model = SGDClassifier(max_iter=1000, random_state=42)
X_test_accum, y_test_accum = [], []


first_batch = True
for X_train_batch, y_train_batch in batch_load_log_files("dataset", bit_len, batch_size=batch_size):
    log.println("processing batch")

    poly = PolynomialFeatures(degree=polynomial_degree, interaction_only=polynomial_interaction_only)
    X_train_transformed = poly.fit_transform(X_train_batch)
    
    if first_batch:
        model.partial_fit(X_train_transformed, y_train_batch, classes=[0, 1])
        first_batch = False
    else:
        model.partial_fit(X_train_transformed, y_train_batch)



log.println("testing in batches")
test_file_path = "dataset"
test_batch_size = 2**16 * 5
accuracies = []
total_samples = 0

for X_test_batch, y_test_batch in batch_load_log_files(test_file_path, bit_len, test_batch_size):
    log.println("process testing batches")
    X_test_transformed = poly.fit_transform(X_test_batch)

    y_pred_batch = model.predict(X_test_transformed)

    batch_accuracy = accuracy_score(y_test_batch, y_pred_batch)
    accuracies.append(batch_accuracy * len(y_test_batch))
    total_samples += len(y_test_batch)

    log.println(f"{accuracies[-1]} / {len(y_test_batch)}")

overall_accuracy = sum(accuracies) / total_samples
log.println(f"Accuracy: {overall_accuracy * 100:.2f}%")
