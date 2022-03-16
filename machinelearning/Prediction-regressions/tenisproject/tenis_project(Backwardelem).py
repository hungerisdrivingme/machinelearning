# -*- coding: utf-8 -*-


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#veri on isleme
"""
boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

"""
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
outlook= veriler.iloc[:,0:1].values
print(outlook)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

outlook[:,0] = le.fit_transform(veriler.iloc[:,0])

print(outlook)

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
windy=veriler.iloc[:,3:4].values
print(windy)


windy[:,0]=le.fit_transform(veriler.iloc[:,3:4])
print(windy)

windy=ohe.fit_transform(windy).toarray()
print(windy)




le = preprocessing.LabelEncoder()
play= veriler.iloc[:,4:5].values
print(play)

play[:,0]=le.fit_transform(veriler.iloc[:,-1:])
print(play)

play=ohe.fit_transform(play).toarray()
print(play)







#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=outlook,  columns = ['sunny','overcast','rainy'])
print(sonuc)

sonuc2 = pd.DataFrame(veriler,  columns = ['temperature','humidity'])
print(sonuc2)





playdf=pd.DataFrame(data=play[:,1],columns=["play"])



windydf = pd.DataFrame(data =windy[:,0] , columns = ["windy"])

print(windydf)

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,windydf], axis=1)
print(s2)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s2,playdf,test_size=0.33, random_state=0)

#verilerin olceklenmesi

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred= regressor.predict(x_test)




# Multiple regression'daki B0 degerlerini ekliyoruz ve Backward Selection yapıyoruz *******
import statsmodels.api as sm


X=np.append(arr=np.ones((14,1)).astype(int), values=s2, axis=1)


X_l= s2.iloc[:,[0,1,2,3,4,5]].values

X_l=np.array(X_l,dtype=float)
model=sm.OLS(playdf,X_l).fit()
print(model.summary())

yenis=s2.drop("temperature",axis=1)



print(yenis)

X1=np.append(arr=np.ones((14,1)).astype(int), values=yenis, axis=1)
X_ll= yenis.iloc[:,[0,1,2,3,4]].values
X_ll=np.array(X_ll,dtype=float)
                 


model2=sm.OLS(playdf,X_ll).fit()
print(model2.summary())

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split


x_train, x_test,y_train,y_test = train_test_split(yenis,playdf,test_size=0.33, random_state=0)

#verilerin olceklenmesi

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y2_pred= regressor.predict(x_test)
