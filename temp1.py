# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 23:06:10 2021

@author: 1832157
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
X_total = np.random.normal(10,1,4000)
X_total = np.reshape(X_total,(1000,4))

    
df1 = pd.read_csv('exp_1_synthetic_supervised_mon.csv', encoding="utf-8",sep=',', header=None)
df2 = pd.read_csv('exp_1_synth_supervised.csv', encoding="utf-8",sep=',', header=None)

features1 = df1.values
features2=df2.values

float_arr1 = np.vstack([features1[i,1:] for i in range(len(features1))]).astype(np.float)
float_arr2 = np.vstack([features2[i,1:] for i in range(len(features2))]).astype(np.float)

y = 0.1*X_total[:,0]+ 0.2*X_total[:,1]+ 0.3*X_total[:,2]+ 0.4*X_total[:,3]
fig, ax = plt.subplots()
sns.kdeplot(x=y, ax = ax,label='original')
sns.kdeplot(x=float_arr1[:,4], ax = ax,label='montonic')
sns.kdeplot(x=float_arr2[:,4], ax = ax,label='supervised')
#sns.kdeplot(x=float_arr2[:,4], ax = ax,label='supervised')

ax.set_xlabel('Score')
ax.set_ylabel('Density')
ax.set_title('Score KDE - Synthetic data set')
#plt.ylim(0, 0.35)
plt.legend()
