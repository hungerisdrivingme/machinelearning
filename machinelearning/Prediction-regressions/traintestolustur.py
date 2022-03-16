# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:47:17 2022

@author: hsyn_
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

#Veri önisleme veri yükleme
veriler=pd.read_csv("eksikveriler.csv")
print(veriler)

Yas=veriler.iloc[:,1:4].values
print(Yas)

# NaN degerleri ortalama yaş ile degiştiriyoruz *********************
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

imputer=imputer.fit(Yas[:,1:4])
(Yas[:,1:4])=imputer.transform(Yas[:,1:4])
print(Yas)

# ulkeleri ayırıyoruz tüm datadan****************
ulke=veriler.iloc[:,0:1].values
print(ulke)

# ulkeleri string ifadeden 1 ve 0 lı ifadeye ceviriyoruz( nominalden ordinale) ******************
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

# Burası da 1 2 0 degerlerini [1 0 1], [0 0 1] yapıyor***********************
ohe=preprocessing.OneHotEncoder()

ulke=ohe.fit_transform(ulke).toarray()
print(ulke)


# Degerleri dataframe cevirdik ******************************************
sonuc=pd.DataFrame(ulke,columns=["fr","tr","usa"])
print(sonuc)

sonuc2=pd.DataFrame(Yas,columns=["boy","kilo","yas"])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
sonuc3=pd.DataFrame(cinsiyet,columns=["cinsiyet"])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)

print(s2)

# Dataları Train ve Test olarak ayırdık %33unu test sectik random atsın diye random stata kullandık
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)

#Degerleri ölçekleyip anlamlı hale getirdik****************************************

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)















