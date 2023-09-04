#############################################################################
# Dependencies, Importing Models and Data
#############################################################################

!pip install transformers;
!pip install torch;
!pip install pandas;
!pip install sklearn;
!install numpy;
# Import necessary libraries
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from transformers import DistilBertTokenizerFast, DistilBertModel
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch
import pandas as pd
import os
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

#
model_paths = [
    ('20200224', '/content/drive/MyDrive/Models/20200224_5_5e7_0.5.pth'),
    ('20200225', '/content/drive/MyDrive/Models/20200225_5_5e7_0.5.pth'),
    ('20200227', '/content/drive/MyDrive/Models/20200227_5_5e7_0.5.pth'),
    ('20200302', '/content/drive/MyDrive/Models/20200302_5_5e7_0.5.pth'),
    ('20200303', '/content/drive/MyDrive/Models/20200303_5_5e7_0.5.pth'),
    ('20200304', '/content/drive/MyDrive/Models/20200304_5_5e7_0.5.pth'),
    ('20200305', '/content/drive/MyDrive/Models/20200305_5_5e7_0.5.pth'),
    ('20200309', '/content/drive/MyDrive/Models/20200309_5_5e6_0.5.pth'),
    ('20200310', '/content/drive/MyDrive/Models/20200310_5_5e7_0.5.pth'),
    ('20200311', '/content/drive/MyDrive/Models/20200311_5_5e7_0.5.pth'),
    ('20200316', '/content/drive/MyDrive/Models/20200316_5_5e7_0.5.pth'),
    ('20200317', '/content/drive/MyDrive/Models/20200317_5_5e7_0.5.pth'),
    ('20200318', '/content/drive/MyDrive/Models/20200318_5_5e6_0.5.pth'),
    ('20200323', '/content/drive/MyDrive/Models/20200323_5_5e7_0.5.pth'),
    ('20200324', '/content/drive/MyDrive/Models/20200324_5_5e7_0.5.pth'),
    ('20200326', '/content/drive/MyDrive/Models/20200326_5_5e7_0.5.pth'),
    ('20200327', '/content/drive/MyDrive/Models/20200327_5_5e7_0.5.pth')
]


#############################################################################
# Redefine custom model and prepare data 
#############################################################################

aggregate_results = pd.DataFrame(columns=['date', 'learning_rate', 'decay_factor', 'r_squared', 'adjusted_r_squared'])

for date, model_path in model_paths:
# model parameters
    BATCH_SIZE = 16
    target = 'ER_1'
    train_companies_path = '/content/drive/MyDrive/Data/Train_Test_Val/train_companies.csv'
    test_companies_path = '/content/drive/MyDrive/Data/Train_Test_Val/test_companies.csv'
    entire_dataset_path = f'/content/drive/MyDrive/Data/512_nonaics/merged_{date}_512.csv'
    map_naics4_naics_path = '/content/drive/MyDrive/Data/map_naics4_naics2.csv'

    # extract learning rate and decay factor from model_path
    learning_rate, decay_factor_with_extension = model_path.split('_')[-2:]
    learning_rate = float(learning_rate)
    decay_factor = float(decay_factor_with_extension.rsplit('.', 1)[0])  # Split by the last '.' and take the first element


    # define the target and structured features
    structured_features = ['lev', 'logEMV']

    # initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # load entire dataset and naics data
    entire_dataset = pd.read_csv(entire_dataset_path)
    map_naics4_naics = pd.read_csv(map_naics4_naics_path)

    entire_dataset = entire_dataset.rename(columns={'tic_x': 'tic'})

    # ensure 'cik' is the same data type in both dataframes
    entire_dataset['cik'] = entire_dataset['cik'].astype(int)
    map_naics4_naics['cik'] = map_naics4_naics['cik'].astype(int)

    # merge datasets on 'cik'
    entire_dataset = pd.merge(entire_dataset, map_naics4_naics, on=['cik'])

    # one hot encoding 'naics2' column
    entire_dataset = pd.get_dummies(entire_dataset, columns=['naics2'])

    # update structured_features
    structured_features += [col for col in entire_dataset.columns if 'naics2' in col]

    # Load CIKs from files for train and test sets
    train_companies = pd.read_csv(train_companies_path)['cik'].unique()
    test_companies = pd.read_csv(test_companies_path)['cik'].unique()

    # create train_data and test_data set from entire_dataset using train_companies and test_companies
    train_data = entire_dataset[entire_dataset['cik'].isin(train_companies)]
    test_data = entire_dataset[entire_dataset['cik'].isin(test_companies)]

    test_data = test_data.copy()
    test_data['text'] = test_data['text'].astype(str)

    # fit scaler on the train_data
    scaler = StandardScaler()
    scaler.fit(train_data[structured_features])

    # Transform the structured_features in test dataset
    test_data.loc[:, structured_features] = scaler.transform(test_data[structured_features])

    # Create cik_tic pairs after splitting the data
    test_data.loc[:, 'cik_tic'] = test_data['cik'].astype(str) + '_' + test_data['tic_x'].astype(str)

    # custom model
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

    # initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertWithStructured()
    model = model.to(device)

    # load model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model weights loaded.")
    else:
        print("No saved model weights found.")



    # tokenize the text data
    inputs = tokenizer(test_data['text'].to_list(), return_tensors='pt', padding=True, truncation=True, max_length=512)

    # prepare DataLoader
    input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).to(device)
    attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).to(device)
    structured_data = torch.tensor(test_data[structured_features].values, dtype=torch.float).to(device)
    targets = torch.tensor(test_data[target].values, dtype=torch.float).to(device)

    test_dataset = TensorDataset(input_ids, attention_mask, structured_data, targets)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # switch to evaluation mode
    model.eval()

    # create a DataFrame to store predictions and targets
    predictions_df = pd.DataFrame(columns=['cik_tic', 'prediction', 'target'])



#############################################################################
# Generating Firm-Level returns predictions
#############################################################################


    # perform predictions and store results
    for i, batch in enumerate(test_dataloader):
        input_ids, attention_mask, structured_data, targets = [b.to(device) for b in batch]

        with torch.no_grad():
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, structured_data=structured_data, labels=targets)
        logits = logits.detach().cpu().numpy().flatten()
        targets = targets.to('cpu').numpy().flatten()

        # add the cik_tic, prediction, and target to the DataFrame
        batch_predictions_df = pd.DataFrame({
            'cik_tic': test_data['cik_tic'][i*BATCH_SIZE:(i+1)*BATCH_SIZE],
            'prediction': logits,
            'target': targets,
        })
        predictions_df = pd.concat([predictions_df, batch_predictions_df])

    # grooup by cik_tic to compute firm-level average prediction and target
    grouped_predictions_df = predictions_df.groupby('cik_tic').mean().reset_index()

    # print the average number of rows per firm
    average_rows_per_firm = len(predictions_df) / len(grouped_predictions_df)
    print(f"Average number of rows per firm: {average_rows_per_firm}")

    # print the number of unique predictions
    unique_predictions = predictions_df['prediction'].nunique()
    print(f"Number of unique predictions: {unique_predictions}")

    # prrint the number of unique firms in final predictions
    num_unique_firms = grouped_predictions_df['cik_tic'].nunique()
    print(f"Number of unique firms in final predictions: {num_unique_firms}")


    # Calculate R-squared and adjusted R-squared
    r_squared = r2_score(grouped_predictions_df['target'], grouped_predictions_df['prediction'])
    adjusted_r_squared = 1 - (1 - r_squared) * (len(grouped_predictions_df['target']) - 1) / (len(grouped_predictions_df['target']) - grouped_predictions_df.shape[1] - 1)

    # these print statements have been modified to include the date
    print(f"Date: {date} | R-squared on test set (firm-level): {r_squared}")
    print(f"Date: {date} | Adjusted R-squared on test set (firm-level): {adjusted_r_squared}")


    aggregate_results = aggregate_results.append({
        'date': date,
        'learning_rate': learning_rate,
        'decay_factor': decay_factor,
        'r_squared': r_squared,
        'adjusted_r_squared': adjusted_r_squared
    }, ignore_index=True)

# anppend the results DataFrame to a CSV file
aggregate_results.to_csv("/content/drive/MyDrive/Data/Metrics/aggregate_results.csv", index=False)
