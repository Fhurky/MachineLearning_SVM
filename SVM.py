# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:36:05 2024

@author: furko
"""

import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, jaccard_score

# Load the dataset
data = pd.read_csv("Cancer_Data.csv")

# Drop the "id" column as it is not useful for the model
data = data.drop(["id"], axis=1)

# Convert the "diagnosis" column to numerical values starting from 0
data["diagnosis"] = pd.factorize(data["diagnosis"])[0]

# Define the target variable Y and the features X
Y = data.iloc[:,0]
X = data.drop(["diagnosis"], axis=1)

# Initialize the StandardScaler and fit it to the features
scaler = StandardScaler()
scaler.fit(X)

# Scale the feature data using the fitted scaler
scaled_Data = scaler.transform(X)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=53)

# Initialize the SVM model with an RBF kernel and a regularization parameter C=0.1
SVM_model = svm.SVC(kernel="rbf", C=0.1)

# Train the SVM model on the training data
SVM_model.fit(x_train, y_train)

# Calculate and print the F1 score for the test set
print("F1_score: : %.2f" % f1_score(y_test, SVM_model.predict(x_test), average='weighted'))

# Calculate and print the Jaccard score for the test set
print("Jaccard_score: : %.2f" % jaccard_score(y_test, SVM_model.predict(x_test), pos_label=0))

# Generate and print the confusion matrix for the test set
print("Confusion_Matrix:\n", confusion_matrix(y_test, SVM_model.predict(x_test), labels=[1,0]))
