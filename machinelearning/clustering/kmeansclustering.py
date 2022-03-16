import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans ( n_clusters = 4, init = 'k-means++')  # kume sayısı ve kmeans++ kullanıldı
kmeans.fit(X)

print(kmeans.cluster_centers_)  # Kmerkezleri nerde olusturdugunu yazar
sonuclar = []
for i in range(1,11):            # k icin optimum degeri bulunur.
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)    #WCSS degeri
plt.show()
# cizime baktıgımızda 2 -4 arasında dirsek noktası görülebilir. Bu sebeple k sayısı 2-4 arasında secilmelidir.

kmeans = KMeans (n_clusters = 4, init='k-means++', random_state= 123)
Y_tahmin=kmeans.fit_predict(X)
print(Y_tahmin)
plt.title("Kmeans")
plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==0,1],color="red")
plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1],color="blue")
plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1],color="yellow")
plt.scatter(X[Y_tahmin==3,0], X[Y_tahmin==3,1],color="green")

plt.show()

