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

data = pd.read_csv("Cancer_Data.csv")
data = data.drop(["id"], axis = 1)

data["diagnosis"] = pd.factorize(data["diagnosis"])[0]

Y = data.iloc[:,0]
X = data.drop(["diagnosis"], axis = 1)

scaler = StandardScaler()
scaler.fit(X)
scaled_Data = scaler.transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=53)

SVM_model = svm.SVC(kernel = "rbf", C=0.1)
SVM_model.fit(x_train, y_train)


# Calculate and print the Log Loss metric
print("F1_score: : %.2f" % f1_score(y_test, SVM_model.predict(x_test), average='weighted') )

# Calculate and print the Jaccard Score for the test set
print("Jaccard_score: : %.2f" % jaccard_score(y_test, SVM_model.predict(x_test), pos_label=0))

# Generate and print the Confusion Matrix
print("Confusion_Matrix:\n", confusion_matrix(y_test, SVM_model.predict(x_test), labels=[1,0]))


