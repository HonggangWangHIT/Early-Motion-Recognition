import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torch.optim import AdamW
import time 
import joblib 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
records = pd.DataFrame(columns=['Filename', 'Epoch', 'Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1'])
res_time =  pd.DataFrame(columns=['Filename', 'Epoch', 'Time Taken'])

data_dir = "../Datasets_F30"
#data_dir = "../Datasets"
#result_path = "../Results/Results_MLP.xlsx"
result_path = "../Results/Results_MLP_F30.xlsx"

file_list = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]

class MLPClassifier(nn.Module):
	def __init__(self):
		super(MLPClassifier, self).__init__()
		self.layer1 = nn.Linear(4, 512)  
		self.layer2 = nn.Linear(512, 256)
		self.output_layer = nn.Linear(256, len(np.unique(y))) 
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.layer1(x))
		x = self.relu(self.layer2(x))
		x = self.output_layer(x)
		return x
        

for file in file_list:
    print(f"Processing file: {file}")
    df = pd.read_excel(os.path.join(data_dir, file))
    
    X = df.iloc[:, 1:5].values  
    y = df.iloc[:, 0].values 
    y = pd.factorize(y)[0] 

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MLPClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            train_preds, train_labels = [], []
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            epoch_loss = sum(epoch_losses) / len(epoch_losses)

            test_preds, test_labels = [], []
            start_time = time.time()
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
            end_time = time.time()  
            time_taken = end_time - start_time  

            train_accuracy = accuracy_score(train_labels, train_preds) * 100
            test_accuracy = accuracy_score(test_labels, test_preds) * 100
            train_precision = precision_score(train_labels, train_preds, average='macro', zero_division=0) * 100
            test_precision = precision_score(test_labels, test_preds, average='macro', zero_division=0) * 100
            train_recall = recall_score(train_labels, train_preds, average='macro', zero_division=0) * 100
            test_recall = recall_score(test_labels, test_preds, average='macro', zero_division=0) * 100
            train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0) * 100
            test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0) * 100

            results_record = pd.DataFrame({
                'Filename': [file] * 1,
                'Epoch': [epoch + 1],
                'Loss': [epoch_loss],
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
            print(f"File: {file}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}, Time taken: {time_taken:.4f}")
            
            if records.empty:
                records = results_record
                res_time = res_time_record
            else:
                records = pd.concat([records, results_record], ignore_index=True)
                res_time = pd.concat([res_time, res_time_record], ignore_index=True)

    #model_save_path = f"../Models/{file.split('.')[0]}_MLP.pkl"
    model_save_path = f"../Models/{file.split('.')[0]}_MLP_F30.pkl"
    joblib.dump(model, model_save_path)  # 保存模型
            


records.to_excel(result_path, index=False)

res_time.to_excel("../Test_time/Time_Taken_MLP_F30.xlsx", index=False)
#res_time.to_excel("../Test_time/Time_Taken_MLP.xlsx", index=False)
print("The training has been completed.")

