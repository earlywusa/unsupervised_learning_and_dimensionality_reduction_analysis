#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 00:34:11 2018

@author: erlisuo
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd

import matplotlib.pyplot as plt


def run_pca(n_c, X_train):
    #apply ICA
    from sklearn.decomposition import PCA
    pca = PCA(n_components = n_c)
    X_train = pca.fit_transform(X_train)
#    X_test = pca.transform(X_test)
#    explained_variance = pca.explained_variance_ratio_
#    print( explained_variance)
    return X_train

def reclassifyTargets(y_values):
    #classify the heat efficiency to 5 categories
    for i in range(0,len(y_values)):
        if(y_values[i] <= 10):
            y_values[i] =0
        elif(y_values[i] <=20):
            y_values[i] = 1
        elif(y_values[i] <=30):
            y_values[i] = 2
        elif(y_values[i] <=40):
            y_values[i] = 3
        else:
            y_values[i] = 4
    return y_values

dataset = pd.read_csv('energy_efficiency.csv')
X_values = dataset.iloc[:,0:8].values
y_values = dataset.iloc[:,8].values

from sklearn.preprocessing import StandardScaler
sc_data = StandardScaler()
X_values = sc_data.fit_transform(X_values)
y_values = reclassifyTargets(y_values)

#X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size = 0.2, random_state = 0)

n_classes = len(np.unique(y_values))

estimator = BayesianGaussianMixture(n_components = n_classes, covariance_type='tied', max_iter=10, random_state = 0)

#estimator.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(n_classes)])

#X_values = run_pca(5, X_values)
estimator.fit(X_values)


y_pred = estimator.predict(X_values)

#train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel())

#y_test_pred = estimator.predict(X_test)
#test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel())
#
group0_pred = []
group1_pred = []
group2_pred = []
group3_pred = []
group4_pred = []
for i in range(0,len(y_pred)):
    if (y_pred[i] == 0):
        group0_pred.append(i)
    elif (y_pred[i] == 1):
        group1_pred.append(i)  
    elif (y_pred[i] == 2):
        group2_pred.append(i)
    elif (y_pred[i] == 3):
        group3_pred.append(i)      
    elif (y_pred[i] == 4):
        group4_pred.append(i) 
        
ogroup0 = []
ogroup1 = []
ogroup2 = []
ogroup3 = []
ogroup4 = []
for i in range(0,len(y_values)):
    if (y_values[i] == 0):
        ogroup0.append(i)
    elif (y_values[i] == 1):
        ogroup1.append(i)  
    elif (y_values[i] == 2):
        ogroup2.append(i)
    elif (y_values[i] == 3):
        ogroup3.append(i)      
    elif (y_values[i] == 4):
        ogroup4.append(i) 

#X1	Relative Compactness 
#X2	Surface Area 
#X3	Wall Area 
#X4	Roof Area 
#X5	Overall Height 
#X6	Orientation 
#X7	Glazing Area 
#X8	Glazing Area Distribution 
#y1	Heating Load 
#y2	Cooling Load
colors = ['navy', 'turquoise', 'darkorange', 'green', 'red']
for n, color in enumerate(colors):
    data = X_values[y_pred == n]
    plt.scatter(data[:,1],data[:,2], color = color)
plt.title('cluster attribute surface area vs wall area')
plt.xlabel('surface area')
plt.ylabel('wall area')    
plt.show()
    
for n, color in enumerate(colors):
    data = X_values[y_pred == n]
    plt.scatter(data[:,0],data[:,3], color = color)
plt.title('cluster attribute relative compactness vs roof area')
plt.xlabel('relative compactness')
plt.ylabel('roof area')   
plt.show()

