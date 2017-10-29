#!/usr/bin/env python
# coding: utf-8

import sys
import sklearn
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, preprocessing, model_selection, tree
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score

# Spam letter classification
# Use the sklearn library to determine whether a letter is spam letter or not.
# python classification.py [R, D, S, N] train.csv test.csv
# You should generate one “predict.csv” every time we execute the script

def lrModel(train_X, train_y):

    lr = linear_model.LogisticRegression(
        penalty='l2', solver='liblinear', multi_class='ovr', verbose=0, n_jobs=1)
    lr.fit(train_X, train_y)
    return lr

def dctModel(train_X, train_y):

    dct = sklearn.tree.DecisionTreeClassifier()
    dct.fit(train_X, train_y)
    return dct

def svcModel(train_X, train_y):
    
    svc = sklearn.svm.SVC(
        C=1, 
        kernel='rbf',
        degree=3,
        gamma='auto',
        decision_function_shape='ovr')
    
    svc.fit(train_X, train_y)
    
    return svc

def mlpModel(train_X, train_y):
    mlp = MLPClassifier(hidden_layer_sizes = (15,), max_iter=2000) 
    mlp.fit(train_X, train_y)
    return mlp


if __name__=='__main__':

    if len(sys.argv) < 2:
        print ('no argument')
        sys.exit()
    elif len(sys.argv) != 4:
        print ('ERROR: # of arguments mismatched')
        sys.exit()
    myargv_tmp = sys.argv
    myargv_tmp.pop(0)
    myargv = myargv_tmp

    train = pd.read_csv('example_train.csv', header=None)
    test = pd.read_csv('example_test.csv', header=None)
    train_X = train.iloc[:, :-1]
    train_y = train.iloc[:,-1]
    test_X = test #test.iloc[:, :-1] #test
    # test_y = #test.iloc[:,-1]

    #dat = pd.read_csv('spambase.csv', header=None)
    #X = dat.iloc[:, :-1]
    #y = dat.iloc[:,-1]
    #train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, y, test_size = 0.1308, shuffle=True)

    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    argv = myargv[0]
    if argv == "R": model = lrModel(train_X, train_y)
    elif argv == "D": model = dctModel(train_X, train_y)
    elif argv == "S": model = svcModel(train_X, train_y)
    elif argv == "N": model = mlpModel(train_X, train_y) # Neural Network
    else: print("INVALID ARGV: " + argv)

    # evaluate
    print(model.score(train_X, train_y))
    loss = -cross_val_score(model, train_X, train_y, cv=5, scoring='neg_mean_squared_error')
    print("Accuracy(neg_mean_squared_error): %0.2f (+/- %0.2f)" % (loss.mean(), loss.std() * 2))

    #print(model.score(test_X, test_y))
    #loss = -cross_val_score(model, test_X, test_y, cv=5, scoring='neg_mean_squared_error')
    #print("Accuracy: %0.2f (+/- %0.2f)" % (loss.mean(), loss.std() * 2))
    predicted = model.predict(test_X)
    ans = []
    for i in range(len(predicted)):
        if predicted[i] < 0.5:
            ans.append(0)
        else:
            ans.append(1)
    # print(ans)
    # print(test_y)

    filename = 'predict.csv'
    my_df = pd.DataFrame(ans)
    my_df.to_csv(filename, index=False, header=False)

