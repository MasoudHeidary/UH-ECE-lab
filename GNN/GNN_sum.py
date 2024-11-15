import random
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import GCNConv, SortAggregation
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from tool.log import Log
import copy
import time

log = Log("log.txt", terminal=True)

# TRAIN=False
batch_size = 4
test_batch_size = 1
model_name = f'GNN-HCI_3_batch{batch_size}_best_model.pth'
print(f"model: {model_name}")

conv_checksum = []
def reset_conv_checksum():
    global conv_checksum
    conv_checksum = []

dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
# train_dataset = dataset.copy()
test_dataset = dataset.copy()
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

if False:
    total_rows, total_cols = 0, 0
    max_rows, max_cols = 0, 0
    num_matrices = 0

    # Iterate through the test loader
    print("Input dimensions (rows x columns) in each batch:")
    for batch in test_loader:
        # Access the 2D feature matrix for the batch
        feature_matrix = batch.x  # Node features
        current_rows, current_cols = feature_matrix.shape
        
        print(f"Batch {num_matrices + 1}: {current_rows} x {current_cols}")
        
        # Update metrics
        total_rows += current_rows
        total_cols += current_cols
        max_rows = max(max_rows, current_rows)
        max_cols = max(max_cols, current_cols)
        num_matrices += 1

    # Calculate average dimensions
    average_rows = total_rows / num_matrices if num_matrices > 0 else 0
    average_cols = total_cols / num_matrices if num_matrices > 0 else 0

    # Print results
    print(f"\nMaximum input dimensions: {max_rows} x {max_cols}")
    print(f"Average input dimensions: {average_rows:.2f} x {average_cols:.2f}")
    exit()

#################### artifical data ####################
if True:
    #note: this will replace the test loader data with artifical data
    ARTIFICIAL_DATA_LEN = 5
    ARTIFICAL_DATA_NAME = "artificial_data.pt"
    ARTIFICAL_DATA_GENERATE = False
    artificial_data_list = []

    def generate_artificial_data(data, min_value=-1000, max_value=1000):
        # Clone the data and modify node features with values outside the -1 to 1 range
        modified_data = copy.deepcopy(data)
        num_nodes, num_features = data.x.size()
        # num_nodes, num_features = 100, 37
        random_values = torch.randint(min_value, max_value+1, (num_nodes, num_features), dtype=torch.float32)
        signs = torch.randint(0, 2, (num_nodes, num_features), dtype=torch.float32) * 2 - 1
        modified_data.x = random_values * signs
        return modified_data

    if ARTIFICAL_DATA_GENERATE:
        for _ in range(ARTIFICIAL_DATA_LEN):
            original_data = random.choice(test_dataset)  # Select a random sample
            artificial_data = generate_artificial_data(original_data)
            artificial_data_list.append(artificial_data)  # Save modified data

        if ARTIFICAL_DATA_NAME:
            torch.save(artificial_data_list, ARTIFICAL_DATA_NAME)
            log.println("artifical data saved")


    artificial_data_list = torch.load(ARTIFICAL_DATA_NAME)
    artificial_loader = DataLoader(artificial_data_list, batch_size=test_batch_size, shuffle=False)
    test_dataset = copy.deepcopy(artificial_data_list)
    test_loader = DataLoader(artificial_data_list, batch_size=test_batch_size, shuffle=False)
    log.println("artifical data loaded")


    if True:
        for index, value in enumerate(test_loader):
            log.println(f"input [{index}]: {value.x.shape}")
        exit()

#################### end ####################


def check_checksum(lst_2, lst_1, atol=1e-6, rtol=1e-5):
    if len(lst_2) != len(lst_1):
        raise RuntimeError("length of lists is not same")
    
    index = []
    for i in range(len(lst_1)):
        if torch.allclose(lst_1[i], lst_2[i], atol=atol, rtol=rtol) != True:
            index.append(i)
    return index

def set_random_weight_to_negative_100(model):
    weights = [param for param in model.parameters() if param.requires_grad]
    selected_weight = random.choice(weights)
    
    flat_weights = selected_weight.view(-1)
    
    random_index = random.randint(0, flat_weights.size(0) - 1)

    with torch.no_grad():
        flat_weights[random_index] = -100

    return model

def set_random_weight_by_percentage(model, percentage=20):
    # weights = [param for param in model.named_parameters() if param[1].requires_grad]
    weights = [
        param for param in model.named_parameters()
        if param[1].requires_grad and "classifier" not in param[0]
    ]
    
    if not weights:
        raise ValueError("No eligible weights found to manipulate.")


    selected_name, selected_weight = random.choice(weights)
    
    flat_weights = selected_weight.view(-1)
    
    random_index = random.randint(0, flat_weights.size(0) - 1)
    
    original_value = flat_weights[random_index].item()

    if abs(original_value) < 0.1:
        new_value = random.random() * random.choice([+1, -1])
        change_type = "R[-1, +1]"
    
    else:
    
        if random.choice([True, False]):
            new_value = original_value * (1 + percentage / 100)
            change_type = "'+'"
        else:
            new_value = original_value * (1 - percentage / 100)
            change_type = "'-'"
    
    with torch.no_grad():
        flat_weights[random_index] = new_value

    # Print information about the change
    log.println(f"[{percentage}%], Parameter '{selected_name}' at index {random_index} was {change_type} from {original_value} to {new_value}")

    return model

# Define the complex model
class Model(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Model, self).__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 1)
        self.sort_pool = SortAggregation(k=30)
        self.conv5 = nn.Conv1d(1, 16, kernel_size=97, stride=97)
        self.conv6 = nn.Conv1d(16, 32, kernel_size=5, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.classifier_1 = nn.Linear(352, 128)
        self.drop_out = nn.Dropout(0.5)
        self.classifier_2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = remove_self_loops(edge_index)

        _sum = torch.full((6,), 0.1)

        x_1 = self.conv1(x, edge_index)
        _sum[0] = torch.sum(x_1)
        x_1 = torch.tanh(x_1)
        
        x_2 = self.conv2(x_1, edge_index)
        _sum[1] = torch.sum(x_2)
        x_2 = torch.tanh(x_2)
        
        x_3 = self.conv3(x_2, edge_index)
        _sum[2] = torch.sum(x_3)
        x_3 = torch.tanh(x_3)
        
        x_4 = self.conv4(x_3, edge_index)
        _sum[3] = torch.sum(x_4)
        x_4 = torch.tanh(x_4)
        
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = self.sort_pool(x, batch)
        x = x.view(x.size(0), 1, x.size(-1))

        x = self.conv5(x)
        _sum[4] = torch.sum(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv6(x)
        _sum[5] = torch.sum(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        out = self.relu(self.classifier_1(x))
        out = self.drop_out(out)
        classes = F.log_softmax(self.classifier_2(out), dim=-1)
        
        conv_checksum.append(_sum)
        # print('.', end='')
        return classes

# Initialize model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(dataset.num_node_features, dataset.num_classes).to(device)
model.load_state_dict(torch.load(model_name, weights_only=True))
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()

# Testing function
def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == data.y).sum().item()
    test_accuracy = correct / len(test_dataset)
    return test_accuracy


torch.set_printoptions(precision=10)

test_accuracy = test()
original_checksum = [x.to('cpu') for x in conv_checksum.copy()]
reset_conv_checksum()
log.println(f'original checksum generated, Test Acc: {test_accuracy:.10f}')



if False:
    start_time = time.time()
    test()
    end_time = time.time()
    elapsed_time = end_time - start_time

    log.println(f"Execution time [1]: {elapsed_time/len(test_dataset):.6f} seconds")
    log.println(f"Execution time [100]: {elapsed_time/len(test_dataset)*100:.6f} seconds")
    log.println(f"Execution time [{len(test_dataset)}]: {elapsed_time:.6f} seconds")


    total_weights = sum(p.numel() for p in model.parameters())
    log.println(f"Total number of weights in the model: {total_weights}")
    exit()


# testing
if True:
    log.println("testing... [running original model twice]")

    test_accuracy = test()
    second_checksum = [x.to('cpu') for x in conv_checksum.copy()]
    reset_conv_checksum()


    log.println(f"original sum [0]: \n{original_checksum[0]}")
    log.println(f"second sum [0]: \n{second_checksum[0]}")
    log.println(f"accu: {test_accuracy:.10f}, failed checksum: {len(check_checksum(original_checksum, second_checksum))}/{len(test_dataset)}")



    # print(f"model manipulation")
    # model = set_random_weight_by_percentage(model)
    # test_accuracy = test()
    # second_checksum = [x.to('cpu') for x in conv_checksum.copy()]
    # reset_conv_checksum()

    # print(f"second sum [0]: \n{second_checksum[0]}")
    # print(f"index checking: {len(check_checksum(second_checksum, original_checksum))} changed out of {len(test_dataset)}")


if True:
    RANGE = 5000

    checksum = []
    original_model = copy.deepcopy(model)

    count_detected_fail = 0
    input_detected_fail = [0 for i in range(len(test_dataset))]

    for i in range(RANGE):
        log.println(f"[{i}]:")

        # manipulate model, run and save checksum
        model = set_random_weight_by_percentage(copy.deepcopy(original_model))
        accu = test()
        checksum.append([_.to('cpu') for _ in conv_checksum.copy()])
        reset_conv_checksum()

        failed_checksum = check_checksum(original_checksum, checksum[-1])
        log.println(f"accu: {accu:.10f}, failed checksum: {len(failed_checksum)}/{len(test_dataset)}")

        if len(failed_checksum) > 0:
            count_detected_fail += 1

            for i in failed_checksum:
                input_detected_fail[i] += 1

            

    log.println(f"detected fails: {count_detected_fail} / {RANGE} [{count_detected_fail/RANGE*100}%]")
    log.println(f"input detected:")
    log.println(f"{input_detected_fail}")


    log.println("sorted input detected")
    sorted_numbers = sorted(enumerate(input_detected_fail), key=lambda x: x[1], reverse=True)
    _print_counter = 0
    for index, value in sorted_numbers:
        if _print_counter <= 100:
            _print_counter += 1
            log.println(f"index: {index}\t:\t{value}")
        else:
            log.f.write(f"index: {index}\t:\t{value}\n")
            log.f.flush()
