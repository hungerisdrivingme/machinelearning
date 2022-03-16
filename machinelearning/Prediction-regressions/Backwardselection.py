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

boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)


#eksik veriler
#sci - kit learn
"""
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)
"""
#encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


c=veriler.iloc[:,-1:].values
print(c)


c[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print(c)

c=ohe.fit_transform(c).toarray()
print(c)




#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(veriler, index = range(22), columns = ['kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c, index = range(22), columns = ["e","k"])
sonuc3=sonuc3.drop("k",1)
print(sonuc3)

sonuc4 = pd.DataFrame(data = boy, index = range(22), columns = ["boy"])

print(sonuc4)

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

"""
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s2,sonuc4,test_size=0.33, random_state=0)

#verilerin olceklenmesi

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred= regressor.predict(x_test)

"""
# Multiple regression'daki B0 degerlerini ekliyoruz ve Backward Selection yapıyoruz *******
import statsmodels.api as sm


X=np.append(arr=np.ones((22,1)).astype(int), values=s2, axis=1)


X_l= s2.iloc[:,[0,1,2,3,4,5]].values

X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonuc4,X_l).fit()
print(model.summary())



X_l= s2.iloc[:,[0,1,2,3,5]].values

X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonuc4,X_l).fit()
print(model.summary())

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s2,sonuc4,test_size=0.33, random_state=0)

#verilerin olceklenmesi

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred= regressor.predict(x_test)