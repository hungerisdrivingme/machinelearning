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

y=veriler.iloc[:,2:]
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


from sklearn.metrics import r2_score

print("Polynomial regression  rsquare degeri")
print(r2_score(Y, lin_reg2.predict(x_poly)))




# En son rasgele bir egitim seviyesinden insan verelim ve kac para alması gerektigini gorelim


print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))


print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))









#### SVR ile birdaha cözdürdük kodu*******************


from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = sc2.fit_transform(Y)



from sklearn.svm import SVR

svr_reg= SVR(kernel="rbf")  #rbf gauss metodu
svr_reg.fit(x_olcekli,y_olcekli)


plt.scatter(x_olcekli,y_olcekli,color="red")

plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")

plt.show()

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))


##Rsquare degeri
from sklearn.metrics import r2_score

print("SVR  rsquare degeri")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))










#DecisionTree**********************************************************

from sklearn.tree import DecisionTreeRegressor

r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X,Y,color="red")

plt.plot(X,r_dt.predict(X))

plt.show()

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

#Rsquare
from sklearn.metrics import r2_score

print("DecisionTree rsquare degeri")
print(r2_score(Y, r_dt.predict(X)))





#RandomForest******************************************************

from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(random_state=0, n_estimators=10)  # estimator agac sayisi 10 secildi

rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color="red")

plt.plot(X, rf_reg.predict(X),color="green")






#Predictionların dogruluk yüzdesi icin r2square metodu kullanılır. 1e ne kadar yakınsa o kadar iyidir

from sklearn.metrics import r2_score

print("Randomforest rsquare degeri")
print(r2_score(Y, rf_reg.predict(X)))


