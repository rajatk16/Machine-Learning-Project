# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:25:54 2019

@author: 14693
"""
# Data Pre-processing

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('3. Data.csv')

print(dataset.isnull().sum())

# Missing Data
dataset["Country"] = dataset["Country"].fillna("Unknown")
dataset["Gender"] = dataset["Gender"].fillna("Not Answered")
print(dataset.isnull().sum())

# Encode categorical data
country = pd.get_dummies(dataset["Country"])
developer = pd.get_dummies(dataset["Developer"])
gender = pd.get_dummies(dataset["Gender"])
dataset_new = pd.concat([country, developer, gender, dataset["Age"], dataset["Purchased"]], axis=1)

# Dependent and Independent
X = dataset_new.iloc[:, 0:-1].values
y = dataset_new.iloc[:, -1].values

# Label encoding Y
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# Splitting the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)