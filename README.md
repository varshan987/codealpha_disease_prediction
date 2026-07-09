# Disease Prediction Model - CodeAlpha Internship 🩺

This repository contains the source code for *Task 4* of the CodeAlpha Machine Learning Internship. The objective is to predict the likelihood of a disease based on patient medical data.

## Project Overview
This project uses the *Pima Indians Diabetes Dataset* to train a classification model. Medical data often contains features measured on vastly different scales (e.g., insulin levels vs. age), so StandardScaler was applied to normalize the dataset. A *Logistic Regression* model was selected, as it is a highly interpretable and standard algorithmic baseline for clinical classification tasks.

## Tech Stack Used
* *Language:* Python
* *Data Manipulation:* pandas, numpy
* *Machine Learning:* scikit-learn (LogisticRegression, StandardScaler)

## Model Performance & Results
The model was evaluated using a 20% test split, achieving the following key metrics:
* *Overall Accuracy:* ~75%
* *ROC-AUC Score:* 0.8147
* *Precision (Class 1):* 0.65
* *Recall (Class 1):* 0.67

## How to Run This Project
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn
