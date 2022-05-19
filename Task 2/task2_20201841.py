# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:44:03 2022

@author: babda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# load the data 
df = pd.read_csv('Regression task data.csv')


# cleaning
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.NaN, strategy = "mean")
imputer = imputer.fit(df.iloc[:,:])
df.iloc[:,:] = imputer.transform(df.iloc[:,:])

#handeling outliers for X
Q1 = df.X.quantile(0.25)
Q3 = df.X.quantile(0.75)
Q2 = df.X.quantile(0.50)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

df_no_outlier_X = df[(df.X>lower_limit)&(df.X<upper_limit)]


#handeling outliers for Y
Q1 = df.Y.quantile(0.25)
Q3 = df.Y.quantile(0.75)
Q2 = df.Y.quantile(0.50)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

df_no_outlier_Y = df[(df.Y>lower_limit)&(df.Y<upper_limit)]



# split columns
y = df_no_outlier_X.iloc[:,-1].values
X = df_no_outlier_Y.iloc[:,0:1].values





# split to train and test
from sklearn.model_selection import train_test_split
X_train,X_test , y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# reshape
X_train= X_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)
X_test= X_test.reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)


# visualization
plt.scatter(X_train, y_train, color='red')
plt.plot(X_test, y_pred_test , color='blue')
plt.title(' pH and bicarbonate Graph  (training set)')
plt.xlabel('pH ')
plt.ylabel(' Bicarbonate ')
plt.show()


plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred_test , color='blue')
plt.title(' pH and bicarbonate Graph(testing set)')
plt.xlabel('pH ')
plt.ylabel(' Bicarbonate ')
plt.show()
