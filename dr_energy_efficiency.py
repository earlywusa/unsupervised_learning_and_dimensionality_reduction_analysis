#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 02:46:11 2018

@author: erlisuo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_pca_variance_ratio(X_train):
    from sklearn.decomposition import PCA
    pca = PCA(n_components = None)
    X_train = pca.fit_transform(X_train)
    explained_variance = pca.explained_variance_ratio_
    return explained_variance

def get_ica_param(n_c,X_train):
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=n_c, max_iter = 1000, tol = 0.1, random_state=0)
    X_train = ica.fit_transform(X_train)

    return ica

def run_ica(n_c, X_train):
    #apply ICA
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=n_c, max_iter = 1000, tol = 0.4, random_state=0)
    X_train = ica.fit_transform(X_train)
    return X_train

def run_pca(n_c, X_train, X_test):
    #apply ICA
    from sklearn.decomposition import PCA
    pca = PCA(n_components = n_c)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    print( explained_variance)
    return [X_train, X_test]

def run_grp(n_c, X_train, X_test, y_train, y_test):
    from sklearn.random_projection import GaussianRandomProjection
    grp = GaussianRandomProjection(n_components = n_c, eps = 0.1)
    X_train = grp.fit_transform(X_train, y_train)
    X_test = grp.transform(X_test)
    return [X_train, X_test]

def run_lda(n_c, X_train, X_test, y_train, y_test):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda = LDA(n_components = n_c)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    return [X_train, X_test]

def get_kurtotic(X_train):
    import scipy
    kurtosis = scipy.stats.kurtosis(a = X_train, fisher = True)
    return kurtosis

dataset = pd.read_csv('energy_efficiency.csv')
X_values = dataset.iloc[:,0:8].values
y_values = dataset.iloc[:,8].values

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
    

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_values)
#X_test = sc.transform(X_test)

from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
#X_test = scaling.transform(X_test)
kurtsis_orig = get_kurtotic(X_train)

kurtsis_s_ica = []
for n_c in range(1,9):
    X_train_ica = run_ica(n_c,X_train)
    kurtsis_ica = get_kurtotic(X_train_ica)
    ab_k = abs(kurtsis_ica)
    print(ab_k)
    kurtsis_s_ica.append(sum(ab_k))
    
    

