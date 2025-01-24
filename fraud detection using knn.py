import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data = {
    'Amount': [100, 200, 150, 300, 500, 1000, 50, 75, 120, 250],
    'Time': [50, 30, 45, 60, 120, 300, 10, 25, 80, 200],
    'Age': [25, 30, 35, 45, 50, 28, 32, 38, 40, 55],
    'Fraud': [0, 0, 0, 1, 1, 0, 0, 0, 1, 0]  
}

df = pd.DataFrame(data)

X = df[['Amount', 'Time', 'Age']]  # Features
y = df['Fraud']  # Labels (Fraud or not)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

root = tk.Tk()
root.title("Credit Card Fraud Detection")
root.geometry("600x400")
root.config(bg="orange")

def predict_fraud():
    try:
        amount = float(amount_entry.get())
        time = float(time_entry.get())
        age = float(age_entry.get())

        input_data = np.array([[amount, time, age]])
        input_data_scaled = scaler.transform(input_data)

        prediction = knn.predict(input_data_scaled)

        if prediction == 1:
            result_label.config(text="Fraudulent Transaction", fg="red")
        else:
            result_label.config(text="Not Fraudulent", fg="green")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

title_label = tk.Label(root, text="Credit Card Fraud Detection", font=("Arial", 20), bg="orange", fg="black")
title_label.pack(pady=20)

amount_label = tk.Label(root, text="Amount", font=("Arial", 14), bg="orange", fg="black")
amount_label.pack()
amount_entry = tk.Entry(root, font=("Arial", 14))
amount_entry.pack(pady=5)

time_label = tk.Label(root, text="Time", font=("Arial", 14), bg="orange", fg="black")
time_label.pack()
time_entry = tk.Entry(root, font=("Arial", 14))
time_entry.pack(pady=5)

age_label = tk.Label(root, text="Age", font=("Arial", 14), bg="orange", fg="black")
age_label.pack()
age_entry = tk.Entry(root, font=("Arial", 14))
age_entry.pack(pady=5)

predict_button = tk.Button(root, text="Predict Fraud", font=("Arial", 14), command=predict_fraud, bg="black", fg="white")
predict_button.pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 16), bg="orange", fg="black")
result_label.pack(pady=20)  

root.mainloop()