#############################################################################
# Dependencies, Importing Data, Setting Hyperparameters
#############################################################################

# install dependencies
!pip install transformers;
!pip install torch;
!pip install pandas;
!pip install sklearn;
!install numpy;


# import dependencies
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch.nn.functional as F

pd.options.mode.chained_assignment = None  # default='warn'

# initialize hyperparameters
LR = 5e-7  # learning rate
DECAY_FACTOR = 0.5  # factor by which the learning rate will be reduced each time.
DECAY_STEP_SIZE = 1  # number of epochs after which the learning rate is reduced.
EPOCHS = 5
BATCH_SIZE = 16
chunksize = 20000

# load entire dataset
entire_dataset = pd.read_csv("PATH_TO_DATASET_HERE")

# insert the paths to train/test companies here
train_companies = pd.read_csv("PATH_TO_TRAIN_COMPANIES")['cik'].unique()
test_companies = pd.read_csv("PATH_TO_TEST_COMPANIES")['cik'].unique()  

# naics4 to naics2 mapping
map_naics4_naics = pd.read_csv("/content/drive/MyDrive/Data/map_naics4_naics2.csv")

# paths to save trained model and its metrics
metrics_path = "PATH_TO_METRICS"  # filename format I used: date_epochs_lr_decay
model_path = 'PATH_TO_MODEL' # filename format I used:date_epochs_lr_decay



#############################################################################
# Custom DistillBer Model
#############################################################################

class DistilBertWithStructured(nn.Module):
    def __init__(self, num_labels=1):
        super(DistilBertWithStructured, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size + len(structured_features), num_labels)

    def forward(self, input_ids=None, attention_mask=None, structured_data=None, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]
        pooled_output = torch.cat((pooled_output, structured_data), dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        return loss, logits
    

# initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')




#############################################################################
# Data Preprocessing
#############################################################################
# define target and structured features
target = 'ER_1'
structured_features = ['lev', 'logEMV']

# ensure 'cik' and 'tic' are the same data type in both dataframes
entire_dataset['cik'] = entire_dataset['cik'].astype(int)
map_naics4_naics['cik'] = map_naics4_naics['cik'].astype(int)
entire_dataset['tic'] = entire_dataset['tic'].astype(str)
map_naics4_naics['tic'] = map_naics4_naics['tic'].astype(str)

# merge datasets on 'cik' and 'tic'
entire_dataset = pd.merge(entire_dataset, map_naics4_naics, on=['cik', 'tic'])

# one hot encoding 'naics2' column
entire_dataset = pd.get_dummies(entire_dataset, columns=['naics2'])

# update structured_features
structured_features += [col for col in entire_dataset.columns if 'naics2' in col]

# take a random sample of the dataset, same as shuffling when frac =1
entire_dataset = entire_dataset.sample(frac= 1.0, random_state=1)

entire_dataset['text'] = entire_dataset['text'].astype(str)
entire_dataset['tic'] = entire_dataset['tic'].astype('category')
entire_dataset['cik'] = entire_dataset['cik'].astype('category')

# any nonstring values in 'text' column?
print(entire_dataset['text'].apply(type).value_counts())

# print out the number of unique companies in each set
print("Number of unique companies in the training set:", len(train_companies))
print("Number of unique companies in the test set:", len(test_companies))

train_data = entire_dataset[entire_dataset['cik'].isin(train_companies)] # train data

# create test_data set from entire_dataset using updated test_companies
test_data = entire_dataset[entire_dataset['cik'].isin(test_companies)]

# fit scaler on the train_data
scaler = StandardScaler()
scaler.fit(train_data[structured_features])

# transform the structured_features in train, validation, and test datasets
train_data[structured_features] = scaler.transform(train_data[structured_features])
test_data[structured_features] = scaler.transform(test_data[structured_features])

print('train data rows: ', train_data.shape[0])
print('test data rows: ', test_data.shape[0])

# free up Memory
del entire_dataset

# initialize the model on the GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print the device being used, should be GPU
print("Using device:", device)
model = DistilBertWithStructured()
model = model.to(device)

# freeze all DistilBert parameters
for param in model.distilbert.parameters():
    param.requires_grad = True

# unfreeze the last layer if you want to 
# for param in model.distilbert.transformer.layer[-1].parameters():
    # param.requires_grad = True

# load the saved model's weights
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model weights loaded.")
else:
    print("No saved model weights found.")

# initialize optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP_SIZE, gamma=DECAY_FACTOR)

# shuffling the training data 
train_data = train_data.sample(frac=1).reset_index(drop=True)

# creating chunks of data in memory
chunks = [train_data[i:i+chunksize] for i in range(0,train_data.shape[0], chunksize)]




#############################################################################
# Training and MEtrics Calculation
#############################################################################


# if metrics file doesn't exist, we create it and add the header
if not os.path.exists(metrics_path):
    with open(metrics_path, 'w') as f:
        f.write("Epoch,Train Loss,Train R^2,Train Adjusted R^2,Test Loss,Test R^2,Test Adjusted R^2\n")

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1} of {EPOCHS}...")

    total_loss = 0
    total_examples = 0
    all_predictions_train = []
    all_targets_train = []

    for chunk_index, chunk in enumerate(chunks):
        # process each chunk
        print(f"Processing chunk {chunk_index}...")

        # load structured features 
        encodings = tokenizer(list(chunk['text']), truncation=True, padding=True)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        targets = torch.tensor(chunk[target].values, dtype=torch.float)
        structured = torch.tensor(chunk[structured_features].values, dtype=torch.float) # structured features

        data = TensorDataset(input_ids, attention_mask, structured, targets) # add structured features to TensorDataset
        dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        model.train()

        # training
        for batch in dataloader:
            input_ids, attention_mask, structured, targets = [b.to(device) for b in batch] # extract structured data

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, structured_data=structured, labels=targets) # pass structured data to model
            loss = outputs[0]
            total_loss += loss.item()
            total_examples += len(targets)  # update total examples count
            all_predictions_train.extend(outputs[1].detach().cpu().numpy())  # store all predictions
            all_targets_train.extend(targets.detach().cpu().numpy())  # store all targets

            loss.backward()
            optimizer.step()

    model.eval()

    # save the updated model's weights after each epoch
    torch.save(model.state_dict(), model_path)

    # compute and print train metrics
    average_train_loss = total_loss / total_examples
    train_r2 = r2_score(all_targets_train, all_predictions_train)
    n = total_examples
    p = len(structured_features) + 768  # number of predictors (structured features + 1 for the text)
    adjusted_r2_train = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)

    print(f"Train loss {average_train_loss}")
    print(f"Train R-squared: {train_r2}")
    print(f"Train Adjusted R-squared: {adjusted_r2_train}")

    # compute and print test metrics
    # convert test data to appropriate format for the model
    encodings = tokenizer(list(test_data['text']), truncation=True, padding=True)
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    targets = torch.tensor(test_data[target].values, dtype=torch.float)
    structured = torch.tensor(test_data[structured_features].values, dtype=torch.float) # structured features

    data = TensorDataset(input_ids, attention_mask, structured, targets) # Add structured features to TensorDataset
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # calculate test set loss and adjusted R-squared

    model.eval()
    predictions = []
    total_test_loss = 0

    for batch in dataloader:
        input_ids, attention_mask, structured, targets = [b.to(device) for b in batch] # extract structured data
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, structured_data=structured, labels=targets) # pass structured data to model
        loss = outputs[0]
        total_test_loss += loss.item()
        predictions.extend(outputs[1].tolist())

    average_test_loss = total_test_loss / len(test_data[target])  # compute average test loss
    test_r2 = r2_score(test_data[target], predictions)
    n = test_data.shape[0]
    p = len(structured_features) + 768  # number of predictors (structured features + 1 for the text)

    print(f"Test loss {average_test_loss}")

    adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)

    print(f"R-squared on test set: {test_r2}")
    print(f"Adjusted R-squared on test set: {adjusted_r2}")

    with open(metrics_path, 'a') as f:
        f.write(f"{epoch+1},{average_train_loss},{train_r2},{adjusted_r2_train},{average_test_loss},{test_r2},{adjusted_r2}\n")

    scheduler.step()
