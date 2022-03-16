# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler =pd.read_csv("sepet.csv",header=None) #kolon isimleri yok
print(veriler)

from apyori import apriori

a=np.array([veriler])

t=[]
for i in range(0,7501):
    t.append([str(veriler.values[i,j])for j in range(0,20)])


kurallar=apriori(t,min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2 )

print(list(kurallar))

#lift confidence sup degerleri girildi. lift 3 secildiginde pepper ve beef birbirini liftliyor

