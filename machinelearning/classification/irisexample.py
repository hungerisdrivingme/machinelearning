
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_excel('iris.xls')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#veri on isleme


#eksik veriler
#sci - kit learn

x=veriler.iloc[:,0:4]
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
print("logisticFunc")
print(cm)


sum=0
for i in range(3):
    sum=sum+cm[i][i]   
print(sum)
a=(cm.sum())
print("logfunc accuracy")
print(sum/a)






#KNN ALGORTHM**********************************
from sklearn.neighbors import KNeighborsClassifier


from sklearn.neighbors import DistanceMetric


knn=KNeighborsClassifier(n_neighbors=5, metric="manhattan")  

knn.fit(X_train,y_train)
y_pred=knn.predict(X_test,)
cm=confusion_matrix(y_test, y_pred)
print("KNN")
print(cm)

 

sum=0
for i in range(3):
    sum=sum+cm[i][i]   
print(sum)
a=(cm.sum())
print("KNN accuracy")
print(sum/a)















# Support Vector Machine ************************(classifier)


from sklearn.svm import SVC

svc= SVC(kernel="linear") 
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)


cm=confusion_matrix(y_test, y_pred)
print("SVM")
print(cm)

sum=0
for i in range(3):
    sum=sum+cm[i][i]   
print(sum)
a=(cm.sum())
print("SVM accuracy")
print(sum/a)





#NaıveBayes **************************

from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)



cm=confusion_matrix(y_test, y_pred)
print("GNB")
print(cm)



sum=0
for i in range(3):
    sum=sum+cm[i][i]   
print(sum)
a=(cm.sum())
print("NaiveBayes accuracy")
print(sum/a)





from sklearn.naive_bayes import MultinomialNB
ml=MultinomialNB()
ml.fit(x_train,y_train)
y_pred=ml.predict(x_test)



cm=confusion_matrix(y_test, y_pred)
print("ML")
print(cm)


sum=0
for i in range(3):
    sum=sum+cm[i][i]   
print(sum)
a=(cm.sum())
print("NaiveBayes Multinomial accuracy")
print(sum/a)






# DecisionTree Classifier *********************************

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion="gini")

dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)


cm=confusion_matrix(y_test,y_pred)
print("DTC")
print(cm)


sum=0
for i in range(3):
    sum=sum+cm[i][i]   
print(sum)
a=(cm.sum())
print("DT classifier accuracy")
print(sum/a)


#RandomForest Classification*******************


from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=5, criterion="gini")

rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print("RFC")
print(cm)


sum=0
for i in range(3):
    sum=sum+cm[i][i]   
print(sum)
a=(cm.sum())
print("RF classifier accuracy")
print(sum/a)


# ROC- TPR - FPR
y_proba=rfc.predict_proba(X_test)  #predict probablity

print(y_proba)

from sklearn import metrics
fpr,tpr,thold=metrics.roc_curve(y_test, y_proba[:,0],pos_label="e")
print(fpr)
print(tpr)


# We approach best accuracy as%98 in 2 diffrent models.
# SVM and RandomForest classfy they both approach %98 accuracy.  
#




