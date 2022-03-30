# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:41:23 2022

@author: hsyn_
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sonar_data=pd.read_csv("Copy of sonar data.csv",header=None) #in last column (Rock or Mine)

print(sonar_data.head(10))
print(sonar_data.shape)
print(sonar_data.describe())
print(sonar_data[60].value_counts())
print(sonar_data.groupby(60).mean())




X=sonar_data.drop(columns=60,axis=1)  #seperate datas
Y=sonar_data[60]



X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.1 ,random_state=1)

model=LogisticRegression()
model.fit(X_train,Y_train)
y_sonuc=model.predict(X_test)
y_sonuc,Y_test



X_train_prediction=model.predict(X_train)    #calculating train accuracy
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print(training_data_accuracy)

X_test_prediction=model.predict(X_test)         #calculating test accuracy
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print(test_data_accuracy)


y_test=np.array(Y_test)   # ı calculate accuracy manually
y_test
t=0
f=0

for i in range (len(Y_test)):
    if y_test[i]==y_sonuc[i]:
        t=t+1
    
    else:
        f=f+1
        
print(t,f)
        
print(t/(t+f))




