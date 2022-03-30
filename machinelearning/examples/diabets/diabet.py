# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:10:08 2022

@author: hsyn_
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


df=pd.read_csv("diabetes.csv")

print(df.shape)
print(df.head(10))
print(df.describe())
print(df["Outcome"].value_counts()) #veri dengesiz dagilmis 500e 268
mean_=(df.groupby("Outcome").mean()) #give us a clue about bloodpressure and bone skin tickness may not necessary
x=df.drop(columns="Outcome",axis=1) 
y=df["Outcome"]






x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#verilerin olceklenmesi


sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)





svc=SVC(kernel="linear") 
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)


cm=confusion_matrix(y_test, y_pred)
print("SVC confussion matrix")
print(cm)



y_pred=svc.predict(X_test)         #calculating test accuracy
test_data_accuracy=accuracy_score(y_pred,y_test)
print(test_data_accuracy)

