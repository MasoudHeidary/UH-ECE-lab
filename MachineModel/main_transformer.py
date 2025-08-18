
"""
WARNING NOTE:
this script is drivin from another main code, and may not follow 
the globalVariable configuration and command line
"""


from globalVariable import *
from tool_d.log import Log
from tool import manipulate, manipualte_percentage

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer, GlueDataTrainingArguments, GlueDataset
from transformers.models.bert.modeling_bert import BertModel, BertEncoder, BertLayer
import os
import random
import time
import math




####################################

log = Log(LOG_FILE_NAME, terminal=True)
Manipulate = False
if DATA_SET != "SST-2":
    raise RuntimeError(f"invalid dataset [{DATA_SET}] on transformer.")

##############################################################################

"""training model"""
def train_SST2():
    class BertSingleEncoderEarlyExit(nn.Module):
        def __init__(self, config, num_labels):
            super().__init__()
            self.num_labels = num_labels

            self.bert_encoder = BertModel.from_pretrained('bert-base-uncased',config=config)

            # Freeze BERT parameters
            #for param in self.bert_encoder.parameters():
            #    param.requires_grad = False

            self.early_exit_classifier = nn.Linear(config.hidden_size, num_labels)
            self.init_weights()

        def init_weights(self):
            nn.init.xavier_normal_(self.early_exit_classifier.weight)
            nn.init.constant_(self.early_exit_classifier.bias, 0.001)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
            outputs = self.bert_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                        position_ids=position_ids, head_mask=head_mask)
            pooled_output = outputs[1]
            logits_early_exit = self.early_exit_classifier(pooled_output)

            return logits_early_exit

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    TASK_NAME = "sst-2"
    DATA_DIR = "./data/SST-2"

    data_args = GlueDataTrainingArguments(
        task_name=TASK_NAME,
        data_dir=DATA_DIR,
    )

    train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")
    print(len(train_dataset))

    batch_size = 8
    def custom_collate(batch):
        input_ids_batch = torch.tensor([item.input_ids for item in batch])
        attention_mask_batch = torch.tensor([item.attention_mask for item in batch])
        labels_batch = torch.tensor([item.label for item in batch])  # Assuming the label attribute exists
        return {'input_ids': input_ids_batch, 'attention_mask': attention_mask_batch, 'labels': labels_batch}

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
    model = BertSingleEncoderEarlyExit(config, num_labels=2)  # Assuming binary classification
    

    print(model)
    #print(model.bert_encoder.encoder.layer[0].attention.self.query.weight)
    #print(model.bert_encoder.encoder.layer[1].attention.self.query.weight)

    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)  # You can adjust the learning rate as needed
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # You can adjust the scheduler parameters as needed

    # Define the loss function (cross-entropy loss)
    loss_fn = nn.CrossEntropyLoss()

    # Define the device (GPU or CPU)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    num_epochs = 10  # You can adjust the number of epochs as needed
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)

            # Compute the loss
            loss = loss_fn(logits, labels)
            #print(loss.data)
            loss.data = loss.data * (-0.05 + 0.1*random.random())
            #print(loss.data)
            #quit(0)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Accumulate the total loss
            total_loss += loss.item()

        # Calculate average loss
        avg_loss = total_loss / len(train_dataloader)

        # Evaluate the model on the validation set
        model.eval()
        total_eval_accuracy = 0.0
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                logits = model(input_ids, attention_mask=attention_mask)

            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions = (predictions == labels).sum().item()
            total_eval_accuracy += correct_predictions

        # Calculate average accuracy
        avg_accuracy = total_eval_accuracy / len(eval_dataset)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Training Loss: {avg_loss:.4f}")
        print(f"  Validation Accuracy: {avg_accuracy:.2%}")

        # Update the learning rate scheduler
        scheduler.step()

    model_save_path = "./modeloutput/fine_tuned_bert_single_encoder_early_exit.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


def test_SST2():
    """testing model"""
    
    class BertSingleEncoderEarlyExit(nn.Module):
        def __init__(self, config, num_labels):
            super().__init__()
            self.num_labels = num_labels
            self.bert_encoder = BertModel.from_pretrained('bert-base-uncased',config=config)

            self.early_exit_classifier = nn.Linear(config.hidden_size, num_labels)
            self.init_weights()

        def init_weights(self):
            nn.init.xavier_normal_(self.early_exit_classifier.weight)
            nn.init.constant_(self.early_exit_classifier.bias, 0.001)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
            outputs = self.bert_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                        position_ids=position_ids, head_mask=head_mask)

            pooled_output = outputs[1]
            logits_early_exit = self.early_exit_classifier(pooled_output)
            return logits_early_exit


    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    TASK_NAME = "sst-2"
    DATA_DIR = "./data/SST-2"
    data_args = GlueDataTrainingArguments(
        task_name=TASK_NAME,
        data_dir=DATA_DIR,
        overwrite_cache=True,
    )

    train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

    batch_size = 8
    def custom_collate(batch):
        input_ids_batch = torch.tensor([item.input_ids for item in batch])
        attention_mask_batch = torch.tensor([item.attention_mask for item in batch])
        labels_batch = torch.tensor([item.label for item in batch])  # Assuming the label attribute exists
        return {'input_ids': input_ids_batch, 'attention_mask': attention_mask_batch, 'labels': labels_batch}

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)

    # Load the tampered model
    tampered_model = BertSingleEncoderEarlyExit(config, num_labels=2)
    model_path = "./modeloutput/fine_tuned_bert_single_encoder_early_exit.pth"
    tampered_model.load_state_dict(torch.load(model_path))
    tampered_model.eval()

    #print(model.bert_encoder.encoder.layer[0].attention.self.query.weight)
    """this is manipulation section for layer to layer"""
    def hook_fn(module, input, output):
        return manipulate(output[0]), *output[1:]
    # Register hook on each encoder layer
    for i, layer in enumerate(tampered_model.bert_encoder.encoder.layer):
        layer.register_forward_hook(hook_fn)


    # Calculate the total number of parameters in the model  
    def count_parameters(model):  
        return sum(p.numel() for p in model.parameters() if p.requires_grad)  

    # Count parameters for the tampered model  
    num_params = count_parameters(tampered_model)    

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tampered_model.to(device)
    total_tampered_accuracy = 0.0
    start_time = time.time()

    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():

            logits = tampered_model(input_ids, attention_mask=attention_mask)

            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions = (predictions == labels).sum().item()
            total_tampered_accuracy += correct_predictions

    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_tampered_accuracy = total_tampered_accuracy / len(eval_dataset)
    print(f"\nEvaluation Accuracy (Tampered Model): {avg_tampered_accuracy:.2%}")
    print(f"\nEvaluation Time: {elapsed_time: .4f} seconds")
    print(f"\nLength of TestData: {len(eval_dataloader)}")
    print(f"Total number of trainable parameters in the model: {num_params}")
    #print(tampered_model)
    eval_size = len(eval_dataset)  
    print(f"Number of samples in evaluation dataset: {eval_size}")  

    # Calculate the number of batches  
    N = math.ceil(eval_size / batch_size)  
    print(f"Number of batches (N): {N}")  


    log.println(f"testing accuracy: {avg_tampered_accuracy:.4f}")
    return avg_tampered_accuracy



if __name__ == "__main__":
    for i in default_manipulate_range:
        manipualte_percentage.set(i/100)
        log.println(f"set manipulate percentage: {i}/100%")
        test_SST2()
        # l.println()
        
    if error_in_time:
        prev_erate, prev_accu = -1, -1
        for t_week, error_rate in enumerate(error_in_time):
            if prev_erate == error_rate:
                # l.println(f"week [{t_week:3}], set error_rate [{error_rate:8.4f}] --skip")
                log.println(f"testing accuracy: {prev_accu :.4f}")
                continue
                
            manipualte_percentage.set(error_rate)
            # l.println(f"week [{t_week:3}], set error_rate [{error_rate:8.4f}]")
            
            prev_erate = error_rate
            prev_accu = test_SST2()
    
    log.println()