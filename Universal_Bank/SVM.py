# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:58:25 2024

@author: furko
"""

import pandas as pd

data = pd.read_csv("UniversalBank.csv")

data = data.drop(["ID"], axis = 1)

Y_personal_loan = data.iloc[:,8:9]
Y_Securities_Account = data.iloc[:,9:10]
Y_CD_account = data.iloc[:,10:11]
Y_Online = data.iloc[:,11:12]
Y_Credit_card = data.iloc[:,12:13]