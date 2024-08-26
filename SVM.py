# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:36:05 2024

@author: furko
"""

import pandas as pd
import sklearn.svm 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix, jaccard_score

data = pd.read_csv("Cancer_Data.csv")
data = data.drop(["id"], axis = 1)

data["diagnosis"] = pd.factorize(data["diagnosis"])[0]