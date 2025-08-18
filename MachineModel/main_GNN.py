
"""
NOTE:
https://github.com/leftthomas/DGCNN?tab=readme-ov-file
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import GCNConv, SortAggregation
from torch.optim import Adam
from sklearn.model_selection import train_test_split

import globalVariable as gvar
from globalVariable import device
from tool_d.log import Log
from tool import manipulate, manipualte_percentage

batch_size = 4
test_batch_size = 1
log = Log(gvar.LOG_FILE_NAME, terminal=True)
model_name = f'./modeloutput/GNN-HCI1_batch{batch_size}.pth'
log.println(f"model: {model_name}")

if gvar.DATA_SET != "NCI1":
    raise RuntimeError(f"invalid dataset [{gvar.DATA_SET}] on GNN.")

# Load the NCI1 dataset
dataset = TUDataset(root='./data/NCI1', name='NCI1')
train_dataset = dataset.copy()
test_dataset = dataset.copy()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

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

        x_1 = self.conv1(x, edge_index)
        x_1 = torch.tanh(x_1)
        x_1 = manipulate(x_1)
        
        x_2 = self.conv2(x_1, edge_index)
        x_2 = torch.tanh(x_2)
        x_2 = manipulate(x_2)
        
        x_3 = self.conv3(x_2, edge_index)
        x_3 = torch.tanh(x_3)
        x_3 = manipulate(x_3)
        
        x_4 = self.conv4(x_3, edge_index)
        x_4 = torch.tanh(x_4)
        x_4 = manipulate(x_4)
        
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = manipulate(x)
        x = self.sort_pool(x, batch)
        x = x.view(x.size(0), 1, x.size(-1))
        x = manipulate(x)

        x = self.conv5(x)
        x = self.relu(manipulate(x))
        x = self.pool(x)
        x = manipulate(x)

        x = self.conv6(x)
        x = self.relu(manipulate(x))
        x = x.view(x.size(0), -1)
        out = self.relu(manipulate(self.classifier_1(x)))
        out = self.drop_out(out)
        classes = F.log_softmax(self.classifier_2(out), dim=-1)
        
        return classes

model = Model(dataset.num_node_features, dataset.num_classes).to(device)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    
    train_accuracy = correct / len(train_dataset)
    return total_loss / len(train_loader), train_accuracy

def test():
    model.load_state_dict(torch.load(model_name, weights_only=True))
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


if __name__ == "__main__":
    if gvar.TRAIN_FLAG:
        if gvar.CONFIRM_TO_TRAIN:
            if(input("run the train mode? (y/n)").lower() not in ['y', 'yes']):
                exit()
        best_accuracy = 0
        epochs = 100

        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = train()
            test_accuracy = test()
            log.println(f'Epoch {epoch:03d}, Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
            
            # Save the best model based on test accuracy
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), model_name)

        log.println(f'Best Test Accuracy: {best_accuracy:.4f}')

    else:
        for i in gvar.default_manipulate_range:
            manipualte_percentage.set(i/100)
            log.println(f"set manipulate percentage: {i}/100%")
            log.println(f"accuracy: {test():.4f}")
            
        if gvar.error_in_time:
            prev_erate, prev_accu = -1, -1
            for t_week, error_rate in enumerate(gvar.error_in_time):
                manipualte_percentage.set(error_rate)
                if prev_erate != error_rate:
                    prev_accu = test()
                prev_erate = error_rate
                log.println(f"testing accuracy: {prev_accu :.4f}")
