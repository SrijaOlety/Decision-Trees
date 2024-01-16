# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:28:43 2023

@author: dell
"""
# importing the data#
import pandas as pd
import numpy as np
df = pd.read_csv("D:\\data science python\\NEW DS ASSESSMENTS\\Fraud_check.csv")
df
df.shape
df.info()

# Creating a  binary target variable based on taxable_income

df['Taxable.Income'] = df['Taxable.Income'].apply(lambda x: 'Risky' if x <= 30000 else 'Good')
df['Taxable.Income']
pd.set_option('display.max_columns', None)
df

# Preprocess the data

# EDA #

#EDA----->EXPLORATORY DATA ANALYSIS
#BOXPLOT AND OUTLIERS CALCULATION #

import seaborn as sns
import matplotlib.pyplot as plt
data = ['City.Population','Work.Experience']
for column in data:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(x=df[column])
    plt.title(" Horizontal Box Plot of column")
    plt.show()
#so basically we have seen the ouliers at once without doing everytime for each variable using seaborn#

"""removing the ouliers"""

import seaborn as sns
import matplotlib.pyplot as plt
# List of column names with continuous variables
continuous_columns = ['City.Population','Work.Experience']

# Create a new DataFrame without outliers for each continuous column
data_without_outliers = df.copy()
for column in continuous_columns:
    Q1 = data_without_outliers[column].quantile(0.25)
    Q3 = data_without_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    data_without_outliers = data_without_outliers[(data_without_outliers[column] >= lower_whisker) & (data_without_outliers[column] <= upper_whisker)]

# Print the cleaned data without outliers
print(data_without_outliers)
df1 = data_without_outliers
df1
# Check the shape and info of the cleaned DataFrame
print(df1.shape)
print(df1.info())

#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
df1.hist()
df1.skew()
df1.kurt()
df1.describe() 


#standardising the data

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df1["Undergrad"] = LE.fit_transform(df1["Undergrad"])
df1["Marital.Status"] = LE.fit_transform(df1["Marital.Status"])
df1["Urban"] = LE.fit_transform(df1["Urban"])
df1["Taxable.Income"] = LE.fit_transform(df1["Taxable.Income"])

df1

# Split the data into features (X) and the target variable (y)

X = df1.drop("Taxable.Income", axis=1)
X
Y = df1["Taxable.Income"]
Y

#Data Partition

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.75,random_state=123)

#Decision Tree

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini',max_depth=9)
DT.fit(X_train,Y_train)
Y_pred_train = DT.predict(X_train)
Y_pred_train
Y_pred_test = DT.predict(X_test)
Y_pred_test

#Metrices

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy Score:",ac1.round(3))  
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Training Accuracy Score:",ac2.round(3))  

#Only runs in Google Colab

from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(DT,filled= True,rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph

print("Number of nodes",DT.tree_.node_count)    
print("Level of depth",DT.tree_.max_depth)  

#Validation Set Approach

training_accuracy = []
test_accuracy = []
for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.75,random_state=i)
    DT = DecisionTreeClassifier(criterion='gini',max_depth=9)
    DT.fit(X_train,Y_train)
    Y_pred_train = DT.predict(X_train)
    Y_pred_test = DT.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

import numpy as np
print("Average Training Accuracy:",np.mean(training_accuracy).round(3)) 
print("Average Test Accuracy:",np.mean(test_accuracy).round(3))


#Random Forest  (Parallel ensemble method)

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(max_depth=9,
                        n_estimators=100,
                        max_samples=0.6,
                        max_features=0.7,
                        random_state=56)    
RF.fit(X_train,Y_train)
Y_pred_train = RF.predict(X_train)
Y_pred_test = RF.predict(X_test)

#Metrices
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy Score:",ac1.round(3)) 
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy Score:",ac2.round(3)) 

#Ada Boost (Adaptive Boosting Technique , Sequential ensemble method)

from sklearn.ensemble import AdaBoostClassifier
ABC = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                         n_estimators=100,
                         learning_rate=0.1)
ABR.fit(X_train,Y_train)
Y_pred_train = ABC.predict(X_train)
Y_pred_test = ABC.predict(X_test)

#Metrices
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy Score:",ac1.round(3))    
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy Score:",ac2.round(3))  
