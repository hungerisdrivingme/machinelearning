# -*- coding: utf-8 -*-

#**************************RANDOM SELECTION**************************
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math



veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random
"""
a=random.randint(0, 10)
print(a)
d=10
N= 10000
toplam=0
secilenler=[]
for n in range(0,N):
    ad=random.randint(0,9)
    odul=veriler.values[n,ad]
    toplam=toplam+odul
    
plt.hist(secilenler)
plt.show()

    
    
#Random kolon seciyor. Puan 1 ise ödüle ekliyor 0sa devam ediyor. Kullanıslı degil

# Bu kötü algoritma bile ödül olarak 1200 ort cıkartıyor. UCB'nin bu ödülü gecmesi gerek

"""


#********************************UCB(UPPER CONFIDIENCE BOUND)  *****************************************

"""
N=10000
d=10
oduller=[0]*d
tiklamalar=[0]*d  # o ana kadarki tıklamalar
toplam=0            #toplam odul
secilenler=[]

for n in range(0,N):
    ad=0        #secilen ilan
    max_ucb=0
    for i in range(0,d):
        if(tiklamalar[i]>0):
            ortalama=oduller[i]/tiklamalar[i]
            delta=math.sqrt(3/2*math.log(n)/tiklamalar[i])
            ucb=ortalama+delta
        else:
            ucb=N*10
            
        if max_ucb<ucb:
            max_ucb=ucb
            ad=i
            
    secilenler.append(ad)
    odul=veriler.values[n,ad]
    toplam=toplam+odul
    tiklamalar[ad]=tiklamalar[ad]+1
    oduller[ad]=oduller[ad]+odul
print("toplam ödül")
print(toplam)

plt.hist(secilenler)
plt.show()
"""
# ***** Thompson Sampling*******************





N=10000
d=10
oduller=[0]*d

toplam=0            #toplam odul
secilenler=[]
birler=[0]*d
sifirlar=[0]*d
for n in range(1,N):
    ad=0        #secilen ilan
    max_th=0
    for i in range(0,d):
        rasbeta=random.betavariate(birler[i]+1,sifirlar[i]+1)
        if rasbeta>max_th:
            max_th=rasbeta
            ad=i
    odul=veriler.values[n,ad]        
    secilenler.append(ad)
    if odul==1:
        birler[ad]=birler[ad]+1
        
    else:
        sifirlar[ad]=sifirlar[ad]+1
    
    toplam=toplam+odul
   
    oduller[ad]=oduller[ad]+odul
print("toplam ödül")
print(toplam)

plt.hist(secilenler)
plt.show()



















