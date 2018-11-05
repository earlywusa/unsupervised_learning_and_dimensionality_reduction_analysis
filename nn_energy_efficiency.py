#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 03:45:52 2018

@author: erlisuo
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

def run_ica(n_c, X_train, X_test):
    #apply ICA
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=n_c,random_state=0)
    X_train = ica.fit_transform(X_train)
    X_test = ica.transform(X_test)
    components = ica.components_
    mixing = ica.mixing_
    iteration = ica.n_iter_
    return [X_train, X_test]

def run_pca(n_c, X_train, X_test):
    #apply ICA
    from sklearn.decomposition import PCA
    pca = PCA(n_components = n_c)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
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

def run_nn(train_accuracy, test_accuracy, X_train, y_train):
    classifier = MLPClassifier(
            hidden_layer_sizes=(700,700,700), 
            max_iter=100,
            alpha=0.001,
            random_state=1
            )
    start_time = time.time()
    classifier.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed time: ", elapsed_time)
    y_pred = classifier.predict(X_test)
    y_pred_train = classifier.predict(X_train)
    score = classifier.score(X_test, y_test)
    score_train = classifier.score(X_train, y_train)
#
    #cm = confusion_matrix(y_test, y_pred)
    print ('energy efficiency mlp: train score: ', score_train,'; test score: ', score)
    train_accuracy.append(score_train)
    test_accuracy.append(score)
    return

dataset = pd.read_csv('energy_efficiency.csv')
X=dataset.iloc[:,0:8].values
y_values=dataset.iloc[:,8].values

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


sc_x = StandardScaler()
X = sc_x.fit_transform(X)



score_tsize_train = []
score_tsize_test = []
#print("for test size from 0.1 to 0.9")

X_train, X_test, y_train, y_test = train_test_split(X, y_values, test_size = 0.2, random_state = 0)

scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train_orig = scaling.transform(X_train)
X_test_orig = scaling.transform(X_test)

X_train = X_train_orig
X_test = X_test_orig
train_accuracy = []
test_accuracy = []

#test code
#result = run_ica(n_c=1, X_train = X_train, X_test = X_test)
#X_train = result[0]
#X_test = result[1]
#
#run_nn(train_accuracy=train_accuracy, test_accuracy = test_accuracy, X_train = X_train, y_train = y_train)

for n_c in range (1, X_train.shape[1]):
    
    #apply GRP

    

#    result = run_ica(n_c=n_c, X_train = X_train_orig, X_test = X_test_orig)
#    result = run_grp(n_c=n_c, X_train = X_train_orig, X_test = X_test_orig, y_train = y_train, y_test = y_test)
    result = run_lda(n_c=n_c, X_train = X_train_orig, X_test = X_test_orig, y_train = y_train, y_test = y_test)
    X_train = result[0]
    X_test = result[1]
    run_nn(train_accuracy=train_accuracy, test_accuracy = test_accuracy, X_train = X_train, y_train = y_train)

    

#    X_train, X_test, y_train, y_test = train_test_split(X, y_1, test_size = 0.2, random_state = 22)
#    
#    
#    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
#    X_train = scaling.transform(X_train)
#    X_test = scaling.transform(X_test)
#    score_tsize_train = []
#    score_tsize_test = []
#    for num_iter in range(3,16,1):
#        maxiter=num_iter*10 
#        classifier = MLPClassifier(
#                hidden_layer_sizes=(700,700,700), 
#                max_iter=maxiter,
#                alpha=0.001,
#                random_state=1
#                )
#        start_time = time.time()
#        classifier.fit(X_train, y_train)
#        elapsed_time = time.time() - start_time
#        print("Elapsed time: ", elapsed_time)
#        y_pred = classifier.predict(X_test)
#        y_pred_train = classifier.predict(X_train)
#        score = classifier.score(X_test, y_test)
#        score_train = classifier.score(X_train, y_train)
#        score_tsize_train.append(score_train)
#        score_tsize_test.append(score)
#        #
#        #cm = confusion_matrix(y_test, y_pred)
#        print ('energy efficiency mlp: train score: ', score_train,'; test score: ', score)