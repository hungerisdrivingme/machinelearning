# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 13:14:35 2022

@author: hsyn_
"""

import pandas as pd

url="https://bilkav.com/satislar.csv"

veriler=pd.read_csv(url)

veriler=veriler.values

X=veriler[:,0:1]
Y=veriler[:,1]

bolme=0.33
from sklearn import model_selection

from sklearn.linear_model import LinearRegression

X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=bolme)

lr=LinearRegression()
lr.fit(X_train,Y_train)
print(lr.predict(X_test))

import pickle

dosya="model.kayit"

pickle.dump(lr,open(dosya,"wb"))   #dosyayı kaydettik

yuklenen=pickle.load(open(dosya,"rb")) #dosyayı okuduk tekrar train etmemize gerek yok zaten etmistik

print(yuklenen.predict(X_test))













