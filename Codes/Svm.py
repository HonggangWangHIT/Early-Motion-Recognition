import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import time
import joblib

dataset_directory = "../Datasets"
#dataset_directory = "../Datasets_F30"

results = []

for filename in os.listdir(dataset_directory):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(dataset_directory, filename)
        data = pd.read_excel(file_path, usecols="A:E")
        X = data.iloc[:, 1:5].values  
        y = data.iloc[:, 0].values   
        y = pd.factorize(y)[0] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = svm.SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr')
        model.fit(X_train, y_train)
        
        start_time = time.time() 
        y_pred_test = model.predict(X_test)
        end_time = time.time() 
        time_taken = end_time - start_time
        
        y_pred_train = model.predict(X_train)

        accuracy_test = accuracy_score(y_test, y_pred_test)* 100.0
        precision_test = precision_score(y_test, y_pred_test, average='macro', zero_division=0)* 100.0
        recall_test = recall_score(y_test, y_pred_test, average='macro', zero_division=0)* 100.0  
        f1_test = f1_score(y_test, y_pred_test, average='macro', zero_division=0)* 100.0

        accuracy_train = accuracy_score(y_train, y_pred_train)* 100.0
        precision_train = precision_score(y_train, y_pred_train, average='macro', zero_division=0)* 100.0
        recall_train = recall_score(y_train, y_pred_train, average='macro', zero_division=0)* 100.0  
        f1_train = f1_score(y_train, y_pred_train, average='macro', zero_division=0)* 100.0

        print(f'filename: {filename}')
        print(f'Test Accuracy: {accuracy_test}')
        print(f'Test Precision: {precision_test}')
        print(f'Train Accuracy: {accuracy_train}')
        print(f'Train Precision: {precision_train}')
        print(f'Time taken: {time_taken:.4f} seconds\n')

        results.append({
            "Filename": filename,
            "Train Accuracy": accuracy_train,
            "Train Precision": precision_train,
            "Train Recall": recall_train,
            "Train F1": f1_train,
            "Test Accuracy": accuracy_test,
            "Test Precision": precision_test,
            "Test Recall": recall_test,
            "Test F1": f1_test,
            "Time Taken": time_taken
        })
        model_save_path = f"../Models/{filename.split('.')[0]}_SVM.pkl"
        #model_save_path = f"../Models/{filename.split('.')[0]}_SVM_F30.pkl"
        joblib.dump(model, model_save_path)
        
results_df = pd.DataFrame(results)
results_df.to_excel("../Results/Results_SVM.xlsx", index=False)
#results_df.to_excel("../Results/Results_SVM_F30.xlsx", index=False)

time_df = pd.DataFrame({"Filename": [result["Filename"] for result in results],
                        "Time Taken (seconds)": [result["Time Taken"] for result in results]})
time_df.to_excel("../Test_time/Time_Taken_SVM.xlsx", index=False)
#time_df.to_excel("../Test_time/Time_Taken_SVM_F30.xlsx", index=False)
print("Have completed training!")
