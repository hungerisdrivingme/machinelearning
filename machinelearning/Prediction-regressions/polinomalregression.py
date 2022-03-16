# -*- coding: utf-8 -*-


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('maaslar.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#dataframe slicing///dilimleme ********************
x=veriler.iloc[:,1:2]
X=x.values      # np degerleri cevirdim. Plotta df bazen problem cıkarabiliyor

y=veriler.iloc[:,-1]
Y=y.values



# linear regression /// Nasıl bir model oldugunu görmek icin yaptık. Deneme yani
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x, y)
plt.plot(x,lin_reg.predict(x))
plt.show()




# Polynomial Regression // Yukarında degerlerin polinom old goruldu. Bu sebeple bu asamaya gecildi

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)  #degree=2 secildi. Bu arttırıldıkca sonucun daha dogru cıkma iht. var

x_poly= poly_reg.fit_transform(X)
print(x_poly)


lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color="yellow")

plt.plot(X,lin_reg2.predict(x_poly),color="black")
plt.show()





# Degree degistirilip tekrar calıstırıldı *******************

poly_reg=PolynomialFeatures(degree=6)  #degree=6 secildi. Bu arttırıldıkca sonucun daha dogru cıkma iht. var

x_poly= poly_reg.fit_transform(X)
print(x_poly)


lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color="yellow")

plt.plot(X,lin_reg2.predict(x_poly),color="black")
plt.show()



# En son rasgele bir egitim seviyesinden insan verelim ve kac para alması gerektigini gorelim


print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))


print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

