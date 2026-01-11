🏦 Loan Approval Prediction using Machine Learning
📌 Project Overview

The Loan Approval Prediction project aims to predict whether a loan application will be approved or rejected based on applicant demographic and financial details. This project demonstrates the complete machine learning workflow, from data preprocessing and exploratory data analysis to model training and evaluation.

The objective is to assist financial institutions in making data-driven and risk-aware lending decisions.

🎯 Problem Statement

Loan approval is a critical process for banks and financial institutions. Manual evaluation can be time-consuming and prone to bias.
This project builds a machine learning model that predicts loan approval status using historical loan applicant data.

📂 Dataset Description

The dataset includes the following key features:
Applicant Income
Co-applicant Income
Loan Amount
Loan Amount Term
Credit History
Gender
Marital Status
Education
Self Employment
Property Area
Target Variable:

🔍 Exploratory Data Analysis (EDA)

Analyzed missing values and handled them appropriately
Studied distribution of numerical features
Visualized categorical variables
Identified relationships between features and loan approval status

🧹 Data Preprocessing

Handling missing values
Encoding categorical variables
Feature scaling (if applicable)
Splitting data into training and testing sets

🤖 Machine Learning Models Used

Logistic Regression
K-Nearest Neighbors
Naive Bayes

Model performance was evaluated using:

Precision Score
Recall
F1 Score
Accuracy Score
Confusion Matrix

📊 Results
Based on the precision score, Naive Bayes emerged as the best-performing model for predicting loan approval.
After applying feature engineering techniques, the Logistic Regression model showed significant improvement in performance. However, both Naive Bayes and Logistic Regression consistently outperformed the K-Nearest Neighbors (KNN) model on this dataset.
These results indicate that probabilistic and linear classification models are more suitable for this particular loan approval dataset compared to distance-based methods like KNN.

🛠️ Technologies & Tools

Programming Language: Python
Libraries:
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
IDE: Jupyter Notebook

📌 Key Learnings

End-to-end machine learning project workflow
Data preprocessing and feature engineering
Model training and evaluation
Real-world financial data handling
