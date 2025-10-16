# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 23:06:10 2021

@author: 1832157
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.read_csv('synthetic.csv', encoding="utf-8",sep=',', header=None)   
    
    
df1 = pd.read_csv('exp_20_synth_unsupervised_bound.csv', encoding="utf-8",sep=',', header=None)
#df2 = pd.read_csv('exp_20_synth_unsupervised_sen.csv', encoding="utf-8",sep=',', header=None)
#df3 = pd.read_csv('exp_20_synth_unsupervised_dbn.csv', encoding="utf-8",sep=',', header=None)

#features3 = df3.values
features1 = df1.values
#features2=df2.values

#float_arr3 = np.vstack([features3[i,1:] for i in range(len(features3))]).astype(np.float)
float_arr1 = np.vstack([features1[i,1:] for i in range(len(features1))]).astype(np.float)
#float_arr2 = np.vstack([features2[i,1:] for i in range(len(features2))]).astype(np.float)

fig, ax = plt.subplots()
sns.kdeplot(x=float_arr1[:,5], ax = ax,label='bound')
#sns.kdeplot(x=float_arr1[:,4], ax = ax,label='sensitivity')
#sns.kdeplot(x=float_arr2[:,4], ax = ax,label='distribution')
#sns.kdeplot(x=float_arr2[:,4], ax = ax,label='supervised')

ax.set_xlabel('Score')
ax.set_ylabel('Density')
ax.set_title('Score KDE - Synthetic data set')
#plt.ylim(0, 0.35)
plt.legend()
