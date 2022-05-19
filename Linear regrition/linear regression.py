# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 09:36:22 2022

@author: babda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
testDataFrame = pd.read_csv('test.csv')
trainDataFrame= pd.read_csv('train.csv')



#take care of numerical values for test
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.NaN, strategy = "mean")
imputer = imputer.fit(testDataFrame.iloc[:,:])
testDataFrame.iloc[:,:] = imputer.transform(testDataFrame.iloc[:,:])

#take care of numerical values for train
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.NaN, strategy = "mean")
imputer = imputer.fit(trainDataFrame.iloc[:,:])
trainDataFrame.iloc[:,:] = imputer.transform(trainDataFrame.iloc[:,:])



#handeling outliers for test
Q1 = testDataFrame.x.quantile(0.25)
Q3 = testDataFrame.x.quantile(0.75)
Q2 = testDataFrame.x.quantile(0.50)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

df_no_outlier_test = testDataFrame[(testDataFrame.x>lower_limit)&(testDataFrame.x<upper_limit)]

#handeling outliers for train
Q1 = testDataFrame.x.quantile(0.25)
Q3 = testDataFrame.x.quantile(0.75)
Q2 = testDataFrame.x.quantile(0.50)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

df_no_outlier_train = testDataFrame[(testDataFrame.x>lower_limit)&(testDataFrame.x<upper_limit)]







# spliting data
X_train = df_no_outlier_train.iloc[:,-1].values
y_train = df_no_outlier_train.iloc[:,0:1].values
X_test = df_no_outlier_test.iloc[:,-1].values
y_test = df_no_outlier_test.iloc[:,0:1].values

# reshape data
X_train= X_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)
X_test= X_test.reshape(-1,1)

# train the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)






# plotting and visualization
plt.scatter(X_train, y_train, color='red')
plt.plot(X_test, y_pred_test , color='blue')
plt.title(' (training set)')
plt.xlabel('horizontal ')
plt.ylabel(' vertical ')
plt.show()


plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred_test , color='blue')
plt.title('  (testing set)')
plt.xlabel('horizontal ')
plt.ylabel(' vertical ')
plt.show()

