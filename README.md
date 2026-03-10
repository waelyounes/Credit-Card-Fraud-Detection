# Credit Card Fraud Detection: An Imbalanced Classification Approach

## Project Overview
This project focuses on building a robust Machine Learning model to detect fraudulent credit card transactions. Fraud detection is a classic "Anomaly Detection" problem characterized by a highly imbalanced dataset, where fraudulent transactions represent a very small fraction of the total data.

## Academic & Professional Context
As a Ph.D. in Applied Mathematics, I have developed this project to demonstrate the practical application of statistical modeling and algorithmic optimization in financial security. This implementation is inspired by the comprehensive ML coursework by Siddhardhan and utilizes the Kaggle Credit Card Fraud Metadata.

## Mathematical & Technical Approach
Dealing with imbalanced data requires more than just standard accuracy metrics. In this project, I implemented:

1.  **Data Preprocessing**: Standardized features using `StandardScaler` to ensure numerical stability for the algorithms.
2.  **Handling Class Imbalance**: Used **Under-sampling** (or SMOTE) to balance the distribution of transactions, ensuring the model doesn't become biased toward legitimate cases.
3.  **Algorithmic Logic**: Implemented **Logistic Regression** as a baseline, leveraging its probabilistic nature to classify transactions.
4.  **Evaluation Metrics**: Focused on **Precision, Recall, and F1-Score** rather than simple accuracy, which is misleading in fraud detection scenarios.

## Key Features
- **Exploratory Data Analysis (EDA)**: Visualizing the distribution of 'Time' and 'Amount' features.
- **Model Training**: Utilizing Scikit-Learn for efficient model deployment.
- **Performance Evaluation**: Detailed classification report and confusion matrix analysis.

## Tools & Libraries
- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation and numerical analysis.
- **Scikit-Learn**: Machine Learning framework.
- **Matplotlib & Seaborn**: Data visualization.

## How to Use
1. Clone the repository.
2. Ensure you have the dataset from Kaggle (`creditcard.csv`).
3. Run the Jupyter Notebook `Fraud_Detection.ipynb`.

---
*Note: This project serves as a bridge between theoretical mathematics and real-world predictive analytics.*
