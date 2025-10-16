# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:33:48 2021

@author: 1832157
"""

import numpy as np
import seaborn as sns
from scipy import stats
def compute_kl_divergence(p_probs, q_probs):
    """"KL (p || q)"""
    kl_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(kl_div)

dib1 = np.random.normal(70,15,1000)
dib2 = np.random.normal(45,15,1000)
dib3 = np.random.exponential(10,1000)

dib3 = dib3+40
sns.kdeplot(dib1)
sns.kdeplot(dib2)
sns.kdeplot(dib3)

'''
hist1 = np.histogram(dib1,bins=10)
hist12 = np.histogram(dib12,bins=10)
hist2 = np.histogram(dib2,bins=10)

sum1=np.sum(dib1)
prob1=dib1/sum1

sum12=np.sum(dib12)
prob12=dib12/sum12

sum2=np.sum(dib2)
prob2=dib2/sum2


print(compute_kl_divergence(prob1,prob2))
print(compute_kl_divergence(prob12,prob2))
print(compute_kl_divergence(prob12,prob1))

print(stats.ks_2samp(dib1,dib2))
print(stats.ks_2samp(dib12,dib2))
print(stats.ks_2samp(dib12,dib1))
'''
