# -*- coding: utf-8 -*-


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#veri on isleme



#eksik veriler
#sci - kit learn

x=veriler.iloc[:,1:4]
y=veriler.iloc[:,4:5]



#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression

logr= LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
print(y_pred)


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, y_pred)
print(cm)

## accuracy= 1/8 almost %13




#KNN ALGORTHM**********************************

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5, metric="minkowski")  #5 komsu ve uzaklık ölcme param minkowski secild
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test,)

cm=confusion_matrix(y_test, y_pred)
print(cm)
 # n 5 iken accuracy 1/8
 



# n_neighbors=1 sayısını 1 e düsürünce algoritma daha dogru sonuc verdi 7/8 basarı

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1, metric="minkowski")  #5 komsu ve uzaklık ölcme param minkowski secild
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test,)

cm=confusion_matrix(y_test, y_pred)
print(cm)

