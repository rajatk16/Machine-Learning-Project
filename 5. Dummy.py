# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:25:54 2019

@author: 14693
"""
# Dummy Classifier

# Import the libraries
import numpy as np
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

# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel='cosine')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# Create the Classifying model
from sklearn.dummy import DummyClassifier
classifier = DummyClassifier(strategy="stratified")

# Train the model
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Accuracy
print("Accuracy of logistic regression classifier on test set: {:.2f}%".format(classifier.score(X_test, y_test)  * 100))

#Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print("The Classifier makes {0} correct predictions and {1} false predictions".format(cm[0][0] + cm[1][1], cm[1][0] + cm[0][1]))
print(classification_report(y_test, y_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10)
accuracies.std()
print("After K-Fold Cross Validation, the classifier's accuracy is {:.2f}%".format(accuracies.mean()*100))
print("The variance of the classifier is {:.2f}%".format(np.sqrt(accuracies.std()*100)))

# Applying Grid Search to find the Best Model and Parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'strategy': ['stratified', 'most_frequent', 'prior', 'uniform']}]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_