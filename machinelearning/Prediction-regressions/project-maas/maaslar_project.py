# -*- coding: utf-8 -*-


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
print(veriler.corr())  # dataların aralarındaki korelasyona bakıldı

veriler=veriler.iloc[:,2:]


x=veriler.iloc[:,:3]
X=x.values
print(x)


y=veriler.iloc[:,3:4]
Y=y.values
print(y)



from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)


import statsmodels.api as sm

model=sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())




from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)  #degree=2 secildi. Bu arttırıldıkca sonucun daha dogru cıkma iht. var

x_poly= poly_reg.fit_transform(X)
print(x_poly)


lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

print("poly ols")
model2=sm.OLS(lin_reg2.predict(x_poly),X)
print(model2.fit().summary())








from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = sc2.fit_transform(Y)



from sklearn.svm import SVR

svr_reg= SVR(kernel="rbf")  #rbf gauss metodu
svr_reg.fit(x_olcekli,y_olcekli)

print("SVR OLS")
model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())




from sklearn.tree import DecisionTreeRegressor

r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


print("DT*****************************")
model4=sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())



from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(random_state=0, n_estimators=10)  # estimator agac sayisi 10 secildi

rf_reg.fit(X,Y.ravel())



print("************FR*****************")
model5=sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())















