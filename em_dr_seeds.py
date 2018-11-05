

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 23:52:20 2018

@author: erlisuo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.mixture import BayesianGaussianMixture

def run_pca(n_c, X_train):
    #apply ICA
    from sklearn.decomposition import PCA
    pca = PCA(n_components = n_c)
    X_train = pca.fit_transform(X_train)
#    X_test = pca.transform(X_test)
#    explained_variance = pca.explained_variance_ratio_
#    print( explained_variance)
    return X_train

dataset = pd.read_csv('seeds.csv')
X = dataset.iloc[:,0:7]
X_values = X.values
y = dataset.iloc[:,7]
y_values = y.values


#X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_values)
#X_test = sc.transform(X_test)

n_classes = len(np.unique(y_values))
estimator = BayesianGaussianMixture(n_components = n_classes, covariance_type='tied', max_iter=100, random_state = 4)

#estimator.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(1,n_classes+1)])
X_train = run_pca(4, X_train)
estimator.fit(X_train)

y_pred = estimator.predict(X_train)

#train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel())

#y_test_pred = estimator.predict(X_test)
#test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel())
group1 = []
group2 = []
group3 = []
for i in range(0, len(y_pred)):
    if(y_pred[i] == 1):
        group1.append(X_train[i])
    elif(y_pred[i] == 2):
        group2.append(X_train[i])
    else:
        group3.append(X_train[i])

colors = ['navy', 'red', 'darkorange']
for n, color in enumerate(colors):
    data = X_train[y_pred == n]
    plt.scatter(data[:,0], data[:,1], marker='x', color=color)
plt.title('cluster attribute f1 and f2')
plt.xlabel('f1')
plt.ylabel('f2')    
#plt.xticks(())
#plt.yticks(())

plt.show()


for n, color in enumerate(colors):
    data = X_train[y_pred == n]
    plt.scatter(data[:,2], data[:,3], marker='x', color=color)
plt.title('cluster attribute f3 and f4')
plt.xlabel('f3')
plt.ylabel('f4')    
#plt.xticks(())
#plt.yticks(())

plt.show()



    
    