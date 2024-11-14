import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import time 
import joblib 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_records = pd.DataFrame(columns=['Filename', 'Epoch', 'Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1'])
res_time =  pd.DataFrame(columns=['Filename', 'Epoch', 'Time Taken'])

data_dir = "../Datasets_F30"
#data_dir = "../Datasets"

all_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hidden = self.gru(x)
        hidden = hidden[-1]
        out = self.classifier(hidden)
        return out

def calculate_metrics(loader, model):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad(): 
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = np.mean(np.array(all_labels) == np.array(all_predictions)) * 100
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)* 100
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)* 100
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)* 100
    
    return accuracy, precision, recall, f1
    
time_steps = 20 

for file in all_files:
    df = pd.read_excel(os.path.join(data_dir, file), usecols="A:E")
    features = df.iloc[:, 1:5].values
    labels = df.iloc[:, 0].values
    labels = pd.factorize(labels)[0] 

    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:i + time_steps])
        y.append(labels[i + time_steps])

    X = np.array(X)
    y = np.array(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    input_dim = 4
    hidden_dim = 1024 
    output_dim = len(torch.unique(y_tensor))  
    num_layers = 4 

    model = GRUClassifier(input_dim, hidden_dim, output_dim, num_layers)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    num_epochs = 128
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(train_loader, model)
        start_time = time.time()
        test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(test_loader, model)
        end_time = time.time()
        time_taken = end_time - start_time
        
        epoch_record = pd.DataFrame({
            'Filename': [file],
            'Epoch': [epoch + 1],
            'Loss': [loss.item()],
            'Train Accuracy': [train_accuracy],
            'Train Precision': [train_precision],
            'Train Recall': [train_recall],
            'Train F1': [train_f1],
            'Test Accuracy': [test_accuracy],
            'Test Precision': [test_precision],
            'Test Recall': [test_recall],
            'Test F1': [test_f1]
        })
        res_time_record = pd.DataFrame({'Filename': [file] * 1, 'Epoch': [epoch + 1], "Time Taken": [time_taken]})
        if all_records.empty:
            all_records = epoch_record
        else:
            all_records = pd.concat([all_records, epoch_record], ignore_index=True)

        if res_time.empty:
            res_time = res_time_record
        else:
            res_time = pd.concat([res_time, res_time_record], ignore_index=True)

        print(f'Dataset: {file}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, '
              f'Time taken: {time_taken:.4f}')
    #model_save_path = f"../Models/{file.split('.')[0]}_GRU_model.pkl"
    model_save_path = f"../Models/{file.split('.')[0]}_GRU_F30.pkl"
    joblib.dump(model, model_save_path)  


results_path = "/root/autodl-tmp/Results/Results_GRU_F30.xlsx"
#results_path = "/root/autodl-tmp/Results/Results_GRU.xlsx"
all_records.to_excel(results_path, index=False)

# 保存时间信息到 Excel 文件
#res_time.to_excel("../Test_time/Time_Taken_GRU.xlsx", index=False)
res_time.to_excel("../Test_time/Time_Taken_GRU_F30.xlsx", index=False)
print("The training has been completed.")

