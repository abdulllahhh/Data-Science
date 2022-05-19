# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:34:59 2022

@author: babda
"""
import pandas as pd
att={
       "Approximate Solar Day":[58.65,243,1,0.4263,0.7458,6.3900],
      "Approximate Diameter":[4880,12102,6792,12000,50800,2320] }
df= pd.DataFrame(att)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.df.max)
print(scaler.data_max_[0])
sdf=scaler.transform(df)

#zscore
import scipy.stats as stats
zscores = stats.zscore(df)

data1 = [1, 3, 4, 5, 7, 9, 2]
 

mean= df.mean(axis=0)

maximum = df.max()
minimum = df.min()
standardDeviation= df.std()


list1=[]

   
# Iterate over the index range from o to max number of columns in dataframe
for i in range (0,(len(df))):
    list1.append(
        (df.iloc[i,0]-mean)/standardDeviation
        )
   
for x in range(len(list1)):
    print(list1[x])
    

list2=[]

for i in range (0,(len(df))):
    list2.append(df.iloc[i,0]/1000)
   
    
list2.clear()    
    
    
    

    
    
    
    
    
    