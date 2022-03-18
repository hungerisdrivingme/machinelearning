# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:27:28 2022

@author: hsyn_
"""
#* Veri önislem*****************************
import pandas as pd
import numpy as np
import matplotlib as plt

veriler=pd.read_csv("Churn_Modelling.csv")




X=veriler.iloc[:,3:13].values
Y=veriler.iloc[:,13].values



from sklearn import preprocessing
le=preprocessing.LabelEncoder()

X[:,1]=le.fit_transform(X[:,1])

X[:,1]=le.fit_transform(X[:,1])

X[:,2]=le.fit_transform(X[:,2])


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ohe=ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],remainder="passthrough")

X=ohe.fit_transform(X)
X=X[:,1:]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

#**********************ANN*****************************


import keras
from keras.models import Sequential  #yapay sinir agları olusturuyor

from keras.layers import Dense         # Layers icin



classifier=Sequential()   #yapay sinir agı olusturuldu

classifier.add(Dense(6,bias_initializer  = 'uniform',  activation="relu",input_dim=11)) 
                        # genelde dense sayısı secilirken output noron sayısı bu ornek icin 1
                        # input noron sayısı yani X=11  2sinin ortalaması alınır (6)
                        #init ile initialize ettik 0'a yakın deger atadık weightlere
                        #act fonk olarak relu sectik
                        #input_dim=11 secildi cünkü X 11 kolonlu giriste 11 nörön var
                        #dense= gizli katmanın icindeki noron sayısı
                        


classifier.add(Dense(6,bias_initializer  = 'uniform',  activation="relu"))  #yeni gizli katman eklendi. inputa gerek yok cunku arada

classifier.add(Dense(1,bias_initializer  = 'uniform',  activation="sigmoid",)) #cıkısta 1 noron var onu yazdım


classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])  
#binary_crossentropy kullandım. Cünkü cıkıs Y 1 ve 0 lardan olusuyor
#sinapsisler üzerindeki degerleri optimize ediyor learningrate vs


classifier.fit(X_train,y_train,epochs=50)  #epochs 50 secildi 50 tur atcak

y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

print(cm)



