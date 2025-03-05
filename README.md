# Credit Risk Classification Project

## Overview
This project uses a machine learning model to classify loans as either healthy or high-risk based on historical lending data. The dataset is provided in `lending_data.csv`, which contains various features related to each loan, including whether the loan status is healthy or high-risk. The goal is to build a model that predicts the loan status and evaluates its performance using different metrics.


Step 1: Import Required Libraries
Start by importing necessary libraries:
- `numpy` and `pandas` for data manipulation.
- `pathlib.Path` for managing file paths.
- `train_test_split` for splitting the dataset into training and testing sets.
- `LogisticRegression` for building the logistic regression model.
- `confusion_matrix` and `classification_report` from `sklearn.metrics` to evaluate the model.

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report


Step 2: Load the Data

Load the dataset lending_data.csv into a Pandas DataFrame.

file_path = Path("../Credit_Risk/Resources/lending_data.csv")
df = pd.read_csv(file_path)
df.head()

Step 3: Prepare the Data

The dataset contains the loan_status column, which indicates whether a loan is healthy (0) or high-risk (1). Separate this column into the labels (y) and the remaining features into the feature set (X).

y = df['loan_status']
X = df.drop(columns='loan_status')

Step 4: Split the Data
Split the data into training and testing sets using the train_test_split function from sklearn.model_selection. Use a random_state of 1 to ensure reproducibility.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)



Step 5: Create and Train the Logistic Regression Model
Create logistic regression model and fit it using the training data (X_train and y_train).

from sklearn.linear_model import LogisticRegression
logistic_regression_model = LogisticRegression(random_state=1)
lr_model = logistic_regression_model.fit(X_train, y_train)


Step 6: Make Predictions
Use the trained logistic regression model to make predictions on both the training and testing datasets.

training_predictions = lr_model.predict(X_train)
testing_predictions = logistic_regression_model.predict(X_test)


Step 7: Evaluate the Model
Confusion Matrix: Generate a confusion matrix to evaluate the model's performance on the testing set.

test_matrix = confusion_matrix(y_test, testing_predictions)
print(test_matrix)

training_report = classification_report(y_train, training_predictions)
print(training_report)



