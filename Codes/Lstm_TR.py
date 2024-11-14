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
records = pd.DataFrame(columns=['Filename', 'Epoch', 'Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1'])
res_time =  pd.DataFrame(columns=['Filename', 'Epoch', 'Time Taken'])

#data_dir = "../Datasets"
#results_path = "../Results/Results_LSTMTR.xlsx"
data_dir = "../Datasets_F30"
results_path = "../Results/Results_LSTMTR_F30.xlsx"

time_steps = 20 

input_dim = 4  
hidden_dim = 800 
output_dim = None 
num_layers = 3 

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
    
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels)) * 100
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
    
    model.train()
    return accuracy, precision, recall, f1

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        x = self.tanh(self.linear1(lstm_output)) 
        x = self.linear2(x)  
        scores = torch.softmax(x, dim=1) 

        weighted_output = scores * lstm_output 
        attended_output = weighted_output.sum(dim=1)  

        return attended_output, scores

class LSTMAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMAttentionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_output, _ = self.lstm(x) 

        attended_output, attention_scores = self.attention(lstm_output)

        out = self.classifier(attended_output)
        return out

for filename in os.listdir(data_dir):
    if filename.endswith(".xlsx"):
        df = pd.read_excel(os.path.join(data_dir, filename), usecols="A:E")

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

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        output_dim = len(torch.unique(y_tensor))

        model = LSTMAttentionClassifier(input_dim, hidden_dim, output_dim, num_layers)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00001)

        num_epochs = 128
        for epoch in range(num_epochs):
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
                'Filename': [filename],
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
            res_time_record = pd.DataFrame({'Filename': [filename] * 1, 'Epoch': [epoch + 1], "Time Taken": [time_taken]})
            
            if records.empty:
                records = epoch_record
                res_time = res_time_record
            else:
                records = pd.concat([records, epoch_record], ignore_index=True)
                res_time = pd.concat([res_time, res_time_record], ignore_index=True)

            print(f'Filename: {filename}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, Time taken: {time_taken:.4f}')
        #model_save_path = f"../Models/{filename.split('.')[0]}_LSTMTR.pkl"
        model_save_path = f"../Models/{filename.split('.')[0]}_LSTMTR_F30.pkl"
        joblib.dump(model, model_save_path)  # 保存模型


records.to_excel(results_path, index=False)

res_time.to_excel("../Test_time/Time_Taken_LSTMTR_F30.xlsx", index=False)
#res_time.to_excel("../Test_time/Time_Taken_LSTMTR.xlsx", index=False)

print("The training has been completed.")
