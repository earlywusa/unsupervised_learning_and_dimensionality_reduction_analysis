#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 23:10:37 2018

@author: erlisuo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def run_pca(n_c, X_train):
    #apply ICA
    from sklearn.decomposition import PCA
    pca = PCA(n_components = n_c)
    X_train = pca.fit_transform(X_train)
#    X_test = pca.transform(X_test)
#    explained_variance = pca.explained_variance_ratio_
#    print( explained_variance)
    return X_train

def draw_kmeans(X_train):
    wcss = []
    n_cluster = 30
    for i in range(1,n_cluster):
        kmeans = KMeans(n_clusters = i, 
                        init = 'k-means++',
                        max_iter = 300,
                        n_init = 10,
                        random_state = 0)
        kmeans.fit(X_train)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,n_cluster), wcss)
    plt.title('kmean K selection')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    return wcss
    
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

def run_kmeans(n_c, X_trans):    
    kmeans = KMeans(n_clusters=n_c, init='k-means++', n_init=10, max_iter = 500, random_state = 0)
    y_kmeans = kmeans.fit_predict(X_trans)
    return y_kmeans

dataset = pd.read_csv('energy_efficiency.csv')
X_values = dataset.iloc[:,0:8].values
y_values = dataset.iloc[:,8].values

from sklearn.preprocessing import StandardScaler
sc_data = StandardScaler()
X_trans = sc_data.fit_transform(X_values)

y_values = reclassifyTargets(y_values)

wcss = draw_kmeans(X_trans)

X_train = run_pca(5, X_trans)
y_train = run_kmeans(n_c=7, X_trans = X_train)

colors = ['navy', 'turquoise', 'darkorange', 'green', 'yellow', 'red', 'purple']
#for n, color in enumerate(colors):
#    data = X_train[iris.target == n]
#    plt.scatter(data[:,0],data[:,1], s=0.8, color = color, label= iris.target_names[n])

for n, color in enumerate(colors):
    data = X_train[y_train == n]
    plt.scatter(data[:,0], data[:,1], s=0.5, color=color)
plt.title('cluster attribute relative compactness and Surface Area ')
plt.xlabel('relative compactness')
plt.ylabel('Surface Area ')      

plt.show()

for n, color in enumerate(colors):
    data = X_train[y_train == n]
    plt.scatter(data[:,1], data[:,3], s=0.5, color=color)
plt.title('cluster attribute Wall Area and Roof Area ')
plt.xlabel('Wall Area')
plt.ylabel('Roof Area ')  

plt.show()

#group0_pred = []
#group1_pred = []
#group2_pred = []
#group3_pred = []
#group4_pred = []
#for i in range(0,len(y_kmeans)):
#    if (y_kmeans[i] == 0):
#        group0_pred.append(i)
#    elif (y_kmeans[i] == 1):
#        group1_pred.append(i)  
#    elif (y_kmeans[i] == 2):
#        group2_pred.append(i)
#    elif (y_kmeans[i] == 3):
#        group3_pred.append(i)      
#    elif (y_kmeans[i] == 4):
#        group4_pred.append(i) 
#        
#        
#group0 = []
#group1 = []
#group2 = []
#group3 = []
#group4 = []
#for i in range(0,len(y_kmeans)):
#    if (y_values[i] == 0):
#        group0.append(i)
#    elif (y_values[i] == 1):
#        group1.append(i)  
#    elif (y_values[i] == 2):
#        group2.append(i)
#    elif (y_values[i] == 3):
#        group3.append(i)      
#    elif (y_values[i] == 4):
#        group4.append(i) 
