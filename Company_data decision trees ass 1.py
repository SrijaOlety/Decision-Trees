# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:34:33 2023

@author: dell
"""

import pandas as pd
import numpy as np
df = pd.read_csv("D:\\data science python\\NEW DS ASSESSMENTS\\Company_Data.csv")
df
df.info()
df.shape

# EDA #

#EDA----->EXPLORATORY DATA ANALYSIS
#BOXPLOT AND OUTLIERS CALCULATION #

import seaborn as sns
import matplotlib.pyplot as plt
data = ['Sales','CompPrice','Income','Advertising','Population','Price','Age','Education']
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
continuous_columns = ['Sales','CompPrice','Income','Advertising','Population','Price','Age','Education']

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
df.hist()
df.skew()
df.kurt()
df.describe() 

# converting the sales column into high and low values #
median_sales = df1["Sales"].median()
df1["Sales"] = ["High" if value > median_sales
               else "Low" for value in df1["Sales"]]
df1["Sales"]

#======================================================================

# standardising the data #

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df1["ShelveLoc"] = LE.fit_transform(df1["ShelveLoc"])
df1["Urban"] = LE.fit_transform(df1["Urban"])
df1["US"] = LE.fit_transform(df1["US"])
"""df1["Sales"] = LE.fit_transform(df1["Sales"])"""
df1

# Split the data into features (X) and the target variable (y)

X = df1.drop("Sales", axis=1)
X
Y = df1["Sales"]
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
    DT = DecisionTreeClassifier(criterion='gini',max_depth=8)
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
RF = RandomForestClassifier(max_depth=10,
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
ABR = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                         n_estimators=100,
                         learning_rate=0.1)
ABR.fit(X_train,Y_train)
Y_pred_train = ABR.predict(X_train)
Y_pred_test = ABR.predict(X_test)

#Metrices
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy Score:",ac1.round(3))    
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy Score:",ac2.round(3))  

#Training Accuracy Score: 1.0
#Test Accuracy Score: 0.745 

#XG Boost (Xtreme Gradient Boosting)


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

# Encode the 'Taxable.Income' column in your entire DataFrame
df1['Sales'] = LE.fit_transform(df1['Sales'])
# Check unique values in the target variable



# Now, 'Good' will be encoded as 0, and 'Risky' will be encoded as 1 in your entire dataset

# Split the data into features (X) and the target variable (Y)
X = df1.drop("Sales", axis=1)
Y = df1["Sales"]
Y
print(Y.unique())

# Re-split the data for modeling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75, random_state=123)

#XG Boost
from xgboost import XGBClassifier
XGBClassifier() # eta=0.001,gamma=10,learning_rate=1,reg_lambda=1,n_estimators=100
# Define and fit the XGBoost model
xgb_model = XGBClassifier(gamma=10, reg_lambda=4, n_estimators=100)
xgb_model.fit(X_train, Y_train)

# Make predictions and evaluate the model as you did before
Y_pred_train = xgb_model.predict(X_train)
Y_pred_test = xgb_model.predict(X_test)
ac1 = accuracy_score(Y_train, Y_pred_train)
ac2 = accuracy_score(Y_test, Y_pred_test)
print("XGB Train Accuracy:", (ac1 * 100.0).round(3))  
print("XGB Test Accuracy:", (ac2 * 100.0).round(3))   


#XG Boost
#Running various models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
XGBClassifier() # eta=0.001,gamma=10,learning_rate=1,reg_lambda=1,n_estimators=100
from sklearn.metrics import accuracy_score

models = []

models.append(('LogisticRegression', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
models.append(('Random Forest Classifier', RandomForestClassifier(max_depth=7)))
models.append(('XGB',XGBClassifier(gamma=10,reg_lambda=4,n_estimators=100))) #eta = 0.01,gamma = 10


for title, modelname in models:
    modelname.fit(X_train, Y_train)
    Y_pred_train = modelname.predict(X_train)
    Y_pred_test = modelname.predict(X_test)
    ac1 = accuracy_score(Y_train,Y_pred_train)
    ac2 = accuracy_score(Y_test,Y_pred_test)
    print(title,"Train Accuracy" , (ac1 * 100.0).round(3))
    print(title,"Test Accuracy",  (ac2 * 100.0).round(3),'\n')









