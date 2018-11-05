#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 00:56:19 2018

@author: erlisuo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

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

def run_kmeans(n_c, X_train, X_test):    
    kmeans = KMeans(n_clusters=n_c, init='k-means++', n_init=10, max_iter = 500, random_state = 0)
    y_pred_train = kmeans.fit_predict(X_train)
    y_pred_test = kmeans.predict(X_test)
    return [y_pred_train, y_pred_test]

dataset = pd.read_csv('seeds.csv')
X = dataset.iloc[:,0:7]
X_values = X.values
y = dataset.iloc[:,7]
y_values = y.values


X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

n_classes = len(np.unique(y_values))

wcss = draw_kmeans(X_train)
y_preds = run_kmeans(n_c = 3, X_train = X_train, X_test = X_test)
y_pred_train = y_preds[0]
y_pred_test = y_preds[1]

colors = ['navy', 'red', 'darkorange']
for n, color in enumerate(colors):
    data = X_train[y_pred_train == n]
    plt.scatter(data[:,0], data[:,1], s=0.8, color=color)
for n, color in enumerate(colors):
    data = X_test[y_pred_test == n]
    plt.scatter(data[:,0], data[:,1], marker='x', color=color)
plt.title('cluster attribute area and perimeter')
plt.xlabel('area')
plt.ylabel('perimeter')    
#plt.xticks(())
#plt.yticks(())

plt.show()


for n, color in enumerate(colors):
    data = X_train[y_pred_train == n]
    plt.scatter(data[:,3], data[:,4], s=0.8, color=color)
for n, color in enumerate(colors):
    data = X_test[y_pred_test == n]
    plt.scatter(data[:,3], data[:,4], marker='x', color=color)
plt.title('cluster attribute length of kernel and width of kernel')
plt.xlabel('length of kernel')
plt.ylabel('width of kernel')    
#plt.xticks(())
#plt.yticks(())

plt.show()

#
for n, color in enumerate(colors):
    data = X_train[y_pred_train == n]
    plt.scatter(data[:,5], data[:,6], s=0.8, color=color)
for n, color in enumerate(colors):
    data = X_test[y_pred_test == n]
    plt.scatter(data[:,5], data[:,6], marker='x', color=color)
plt.title('cluster attribute asymmetry coefficient and length of kernel groove')
plt.xlabel('asymmetry coefficient')
plt.ylabel('length of kernel groove')    
#plt.xticks(())
#plt.yticks(())

plt.show()