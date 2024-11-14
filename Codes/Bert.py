import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
import time
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
records = pd.DataFrame(columns=['Filename', 'Epoch', 'Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1'])
res_time =  pd.DataFrame(columns=['Filename', 'Epoch', 'Time Taken'])

#data_dir = "../Datasets"
data_dir = "../Datasets_F30"

#tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('../Models/bert-base-chinese',local_files_only=True)

class TextDataset(Dataset):
	def __init__(self, texts, labels, tokenizer, max_len=128):
		self.texts = texts
		self.labels = labels
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, idx):
		tokenized_data = self.tokenizer(self.texts[idx], max_length=self.max_len,
										 padding='max_length', truncation=True, return_tensors="pt")
		return {
			'input_ids': tokenized_data['input_ids'].squeeze(0),  # Squeeze to remove batch dimension
			'attention_mask': tokenized_data['attention_mask'].squeeze(0),
			'labels': torch.tensor(self.labels[idx], dtype=torch.long)
		}

    
for filename in os.listdir(data_dir):
    if filename.endswith(".xlsx"):
        df = pd.read_excel(os.path.join(data_dir, filename))
        df['text'] = df.iloc[:, 1:5].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        X = df['text'].values
        y = df.iloc[:, 0].values 
        y = pd.factorize(y)[0] 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        train_dataset = TextDataset(X_train, y_train, tokenizer)
        test_dataset = TextDataset(X_test, y_test, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        model = BertForSequenceClassification.from_pretrained('/root/autodl-tmp/Models/bert-base-chinese', num_labels=len(set(y)), local_files_only=True)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=1e-5)

        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            epoch_losses = []
            for batch in tqdm(train_loader, desc=f'Training {filename}, Epoch {epoch + 1}'):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                all_preds, all_labels = [], []
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    all_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())

                epoch_loss = sum(epoch_losses) / len(epoch_losses)
                train_acc = accuracy_score(all_labels, all_preds) * 100
                train_prec = precision_score(all_labels, all_preds, average='macro') * 100
                train_rec = recall_score(all_labels, all_preds, average='macro') * 100
                train_f1 = f1_score(all_labels, all_preds, average='macro') * 100
            
                all_test_preds, all_test_labels = [], []
                start_time = time.time() 
                for batch in test_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    all_test_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                    all_test_labels.extend(batch['labels'].cpu().numpy())

                end_time = time.time() 
                time_taken = end_time - start_time 
                test_acc = accuracy_score(all_test_labels, all_test_preds) * 100
                test_prec = precision_score(all_test_labels, all_test_preds, average='macro', zero_division=0) * 100
                test_rec = recall_score(all_test_labels, all_test_preds, average='macro', zero_division=0) * 100
                test_f1 = f1_score(all_test_labels, all_test_preds, average='macro', zero_division=0) * 100
            
                print(f"File: {filename}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}, Time taken: {time_taken:.4f} seconds\n")

                results_record = pd.DataFrame({
                    'Filename': [filename] * 1,
                    'Epoch': [epoch + 1],
                    'Loss': [epoch_loss],
                    'Train Accuracy': [train_acc],
                    'Train Precision': [train_prec],
                    'Train Recall': [train_rec],
                    'Train F1': [train_f1],
                    'Test Accuracy': [test_acc],
                    'Test Precision': [test_prec],
                    'Test Recall': [test_rec],
                    'Test F1': [test_f1]
                })
                res_time_record = pd.DataFrame({'Filename': [filename] * 1, 'Epoch': [epoch + 1], "Time Taken": [time_taken]})
            
                if records.empty:
                    records = results_record
                else:
                    records = pd.concat([records, results_record], ignore_index=True)
                
                if res_time.empty:
                    res_time = res_time_record
                else:
                    res_time = pd.concat([res_time, res_time_record], ignore_index=True)
        #model_save_path = f"../Models/{filename.split('.')[0]}_BERT_model.pkl"
        model_save_path = f"../Models/{filename.split('.')[0]}_BERT_F30.pkl"
        joblib.dump(model, model_save_path) 

#records.to_excel("../Results/Results_BERT.xlsx", index=False)
records.to_excel("../Results/Results_BERT_F30.xlsx", index=False)

#res_time.to_excel("../Test_time/Time_Taken_BERT.xlsx", index=False)
res_time.to_excel("../Test_time/Time_Taken_BERT_F30.xlsx", index=False)
print("The training has been completed.")
