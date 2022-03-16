# -*- coding: utf-8 -*-


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

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
    
    
    
#Random kolon seciyor. Puan 1 ise ödüle ekliyor 0sa devam ediyor. Kullanıslı degil

# Bu kötü algoritma bile ödül olarak 1200 ort cıkartıyor. UCB'nin bu ödülü gecmesi gerek
