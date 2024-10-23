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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

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
        return r_bin(A), r_bin(B), result
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
        X_batch.append([A, B, fa_i, fa_j, t_index])
        y_batch.append(1 if result else 0)

        if len(X_batch) == batch_size:
            yield np.array(X_batch), np.array(y_batch)
            X_batch, y_batch = [], []

    # Yield remaining data
    if X_batch:
        yield np.array(X_batch), np.array(y_batch)



# batch_size = 2**16
# polynomial_degree = 2
# polynomial_interaction_only = True

# model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
# X_test_accum, y_test_accum = [], []


# first_batch = True
# for X_train_batch, y_train_batch in batch_load_log_files("dataset", bit_len, batch_size=batch_size):
#     log.println("processing batch")

#     poly = PolynomialFeatures(degree=polynomial_degree, interaction_only=polynomial_interaction_only)
#     X_train_transformed = poly.fit_transform(X_train_batch)
    
#     if first_batch:
#         model.partial_fit(X_train_transformed, y_train_batch, classes=[0, 1])
#         first_batch = False
#     else:
#         model.partial_fit(X_train_transformed, y_train_batch)



# log.println("testing in batches")
# test_file_path = "dataset"
# test_batch_size = batch_size
# accuracies = []
# total_samples = 0

# for X_test_batch, y_test_batch in batch_load_log_files(test_file_path, bit_len, test_batch_size):
#     log.println("process testing batches")
#     X_test_transformed = poly.fit_transform(X_test_batch)

#     y_pred_batch = model.predict(X_test_transformed)

#     batch_accuracy = accuracy_score(y_test_batch, y_pred_batch)
#     accuracies.append(batch_accuracy * len(y_test_batch))
#     total_samples += len(y_test_batch)

#     log.println(f"{accuracies[-1]} / {len(y_test_batch)}")

# overall_accuracy = sum(accuracies) / total_samples
# log.println(f"Accuracy: {overall_accuracy * 100:.2f}%")




batch_size = 2**16 * 1
n_components_pca = 5  # Reduce to 50 principal components

# Set up model (Random Forest with class balancing)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Training in batches
X_test_accum, y_test_accum = [], []
first_batch = True

for X_train_batch, y_train_batch in batch_load_log_files("dataset", bit_len, batch_size=batch_size):
    log.println("processing training batch")

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components_pca)
    X_train_transformed = pca.fit_transform(X_train_batch)

    # Fit model on the current batch
    model.fit(X_train_transformed, y_train_batch)

log.println("testing in batches")

# Testing in batches
total_samples = 0
total_correct_predictions = 0

for X_test_batch, y_test_batch in batch_load_log_files("dataset", bit_len, batch_size=batch_size):
    log.println("processing test batch")
    
    # Apply the same PCA transformation
    X_test_transformed = pca.transform(X_test_batch)
    
    y_pred_batch = model.predict(X_test_transformed)

    # Count correct predictions for the batch
    correct_predictions = np.sum(y_pred_batch == y_test_batch)
    total_correct_predictions += correct_predictions
    total_samples += len(y_test_batch)

    log.println(f"Processed batch with {len(y_test_batch)} samples, correct predictions: {correct_predictions}")

# Calculate overall accuracy
accuracy = total_correct_predictions / total_samples if total_samples > 0 else 0
log.println(f"Total samples: {total_samples}")
log.println(f"Total correct predictions: {total_correct_predictions}")
log.println(f"Accuracy: {accuracy * 100:.2f}%")