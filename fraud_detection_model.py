import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- Step 1: Data Loading ---
# Load the dataset (Assuming the file is in the same directory)
# Dataset source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
try:
    credit_card_data = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found. Please download it from Kaggle.")

# --- Step 2: Data Exploration & Preprocessing ---
# Separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Statistical measures of the data
print(f"Legit transactions: {legit.shape}")
print(f"Fraudulent transactions: {fraud.shape}")

# --- Step 3: Handling Imbalanced Data (Under-sampling) ---
# Building a sample dataset containing similar distribution of both classes
# We take a random sample from legit transactions equal to the number of fraud cases
legit_sample = legit.sample(n=492, random_state=2)

# Concatenating two DataFrames
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# --- Step 4: Feature Scaling & Splitting ---
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Standardizing the features (Essential for Logistic Regression convergence)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# --- Step 5: Model Training (Logistic Regression) ---
# Chosen for its probabilistic interpretation in binary classification
model = LogisticRegression()
model.fit(X_train, Y_train)

# --- Step 6: Model Evaluation ---
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print(f'Accuracy on Training data: {training_data_accuracy:.2f}')
print(f'Accuracy on Test data: {test_data_accuracy:.2f}')

# --- Step 7: Detailed Performance Metrics ---
# Since accuracy is misleading in imbalanced datasets, we check Precision and Recall
print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, X_test_prediction))
