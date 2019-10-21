#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep neural network for hard drives failure prediction

Created on Thu Oct 17 11:30:17 2019

@author: Ephraim Iorkyase, ephraim.iorkyase@engineer.com
"""
"""
This is a script for training and evaluating a deep neural network model for detecting/predicting hard drives..

Data source: https://www.kaggle.com/blackblaze/hard-drive-test-data
Data dates: Jan, 2016 - April 2016

The script reads in a CSV with the data.

The script performs k-fold cross-validation and then test the model on 30% of unseen data.

We assume two hidden layers, the number of neurons per layer is set
to approximately half the number of features as is standard practice

"""
#import libraries
import gc
import math
import random
from random import shuffle, randrange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import h5py as h5
from sklearn.externals.joblib import dump, load

# model parameters
NUM_REDUCED_FEATURES = 10 # number of features to select
NUM_VALIDATION_FOLDS = 5  # number of cross-validation folds to use in model metrics
PREDICTION_THRESHOLD = 0.5 # threshold to apply to predictions for metrics
NO_HIDDEN_LAYERS = 2   # no. of hidden layers to include in model (units = NUM_REDUCED_FEATURES//2+1)
DROPOUT = 0.1       # dropout value to apply, False to not apply
# PART 1: Data preprocessing
# import data
datapathname = '/Users/ephraim/Downloads/harddrive.csv' # you can use the path where your data is stored on your computer

df = pd.read_csv(datapathname)
    
#   view data
df.head()
df['model'].value_counts().shape
df.groupby('model')['failure'].sum().sort_values(ascending=False).iloc[:30]

#   from the data viewed, it shows that there are 69 drive models, out of which ST4000DM000 contains 
#   majority of failed drives. It is good practice to compare different drives since reported stat for the same SMART stat can vary in meaning based 
#   on the drive manufacturer and the model. Therefore for the purpose of this analysis I decided to select the drive model with enough data, that is 
#   ST4000DM000. From information provided, 53% of data is ST4000DM000 model.
    
#   select ST4000DM000 model data
df_st4 = df.query('model == "ST4000DM000"')
del df
gc.collect()
    
#   Remove normalised features since both raw & normalised features represent same data points. Moreover, how
#   normalisation is done is not clearly defined. I will carry out any normalisation/standardisation if needed
for col in df_st4.columns:
    if 'normalized' in col:
        del df_st4[col]
gc.collect()

#   drop columns with NAN and constant columns 
df_st4 = df_st4.dropna(axis = 1) 
for i in df_st4.columns:
    if len(df_st4.loc[:,i].unique()) == 1:
        df_st4.drop(i, axis = 1, inplace = True)

# Prepare data for training
# aggregate data to compute number of failed and not failed drives and reduce data size accordingly
df_st4aggregate = (df_st4[['date','serial_number','failure']].groupby('serial_number', as_index = True)
                   .agg({'date':'count','failure':'sum'})
                   .rename(columns={'date':'date_count','failure':'failure_sum'})
                   .sort_values(by=['failure_sum'], axis=0, ascending=False))

# some drives failed more than once. Therefore drop all duplicates
df_st4  = df_st4.drop_duplicates()

# select data on the day of failure (last day)
df_st4_group = df_st4.groupby('serial_number')
df_st4lastday = df_st4_group.nth(-1)
del df_st4
gc.collect()

# drop date and reset index
df_st4lastday=df_st4lastday.drop(['date'], axis=1)
df_st4lastday = df_st4lastday.reset_index(drop = True)

#Count data
df_st4lastday['failure'].value_counts()
df_st4lastday['failure'].value_counts().plot.bar()
print('proportion of the classes in the data:')
print(df_st4lastday['failure'].value_counts()/len(df_st4lastday)) # you can chose to view the proportion of classes

# now we have our data ready for building a model but it is an imbalanced data and we have to deal with this.
# Convert to numpy array and divide to separate input and output data
df_X = df_st4lastday.drop(columns=['failure'])
df_Y = df_st4lastday.iloc[:,0]
X = np.array(df_X)
Y = np.array(df_Y)

# Need to scale features to meet normal distribution with
# mean=0 and sigma=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Now apply recursive feature elimination  
model = LogisticRegression(solver='lbfgs', 
                           max_iter = 10000) # use simple logistic regression model
rfe = RFE(model, n_features_to_select=NUM_REDUCED_FEATURES)
fit = rfe.fit(X_scaled, Y)
reduced_columns = df_X.columns[fit.support_] # you can chose the see the selected features
print(reduced_columns)

# save reduced scaler for use in predictor
reduced_features = df_X[reduced_columns]
reduced_scaler = StandardScaler()
reduced_features_scaled = reduced_scaler.fit_transform(np.array(reduced_features))
#joblib.dump(reduced_scaler,
 #           'classifiers/reduced_scaler_{}hidden_{}random.h5'.format(NO_HIDDEN_LAYERS,
 #                                                        NO_RANDOM_FEATURES))

# Now we have selected 10 best features.
# To deal with imbalanced data, first split data into training and testing data.
X_train, X_test, y_train, y_test = train_test_split(reduced_features_scaled, Y, 
                                                    test_size = 0.33, random_state = 2, shuffle = True, stratify = Y)  

# use SMOTE to produce balanced training data
sm = SMOTE(random_state = 33)  
X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel())

#check the balanced data?
pd.Series(y_train_new).value_counts() # you can chose to check

# perform k-fold iteration and save accuracy and recall values
kfold = StratifiedKFold(n_splits=NUM_VALIDATION_FOLDS, shuffle=True, random_state=2017)
iteration = 1
accscores = []
recallscores = []
au_rocscores = []
for train, test in kfold.split(X_train_new, y_train_new):
    num_inputs = X_train_new.shape[1]
    
# define and fit the model
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = NUM_REDUCED_FEATURES//2+1,
                         kernel_initializer = 'uniform',
                         activation = 'relu',
                         input_dim = num_inputs))
    
    for i in np.arange(NO_HIDDEN_LAYERS-1):
        # Add hidden layer
        classifier.add(Dense(units = NUM_REDUCED_FEATURES//2+1,
                             kernel_initializer = 'uniform',
                             activation = 'relu'))
    
    if DROPOUT:
        classifier.add(Dropout(DROPOUT)) # to avoid overfitting

    # Adding the output layer
    classifier.add(Dense(units = 1,
                         kernel_initializer = 'uniform',
                         activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam',
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy'])
    #class_weights = class_weight.compute_class_weight('balanced',
    #                                                  np.unique(Y[train]),
    #                                                  Y[train])

    # Fitting the ANN to the Training set
    classifier.fit(X_train_new[train],
                   y_train_new[train],
                   batch_size = 10,
                   epochs = 50,
    #               class_weight=class_weights,
                   verbose=0)
    #classifier.save('classifiers/classifier_{}hidden_{}random.h5'.format(NO_HIDDEN_LAYERS,
      #                                                       NO_RANDOM_FEATURES))

    # Making predictions and evaluating the model
    # Predicting the Test set results
    Y_pred = classifier.predict(X_train_new[test])
    Y_pred_thres = (Y_pred > PREDICTION_THRESHOLD)
    #prob_y =classifier.predict_proba(reduced_features_scaled[test])
    #keep only positive class
    prob_y=[p[0] for p in Y_pred_thres]
    #AUROC
    au_roc = roc_auc_score(y_train_new[test],Y_pred)
    # Making the Confusion Matrix
    cm = confusion_matrix(y_train_new[test], Y_pred_thres)
    cr = classification_report(y_train_new[test],Y_pred_thres)

    recall = cm[1,1] / (cm[1,1] + cm[1,0]) * 100
    acc = (cm[0,0] + cm[1,1])/ (cm[0,0] + cm[1,1] + cm[0,1] + cm[1,0]) *100
    accscores.append(acc)
    recallscores.append(recall)
    au_rocscores.append(au_roc)

    iteration += 1

# output mean accuracy/recall
print("Accuracy Ave = " , np.mean(accscores)) # you can chose to print these ones or not
print("Accuracy Std = " , np.std(accscores))
print("Recall Ave = " , np.mean(recallscores))
print("Recall Std = " , np.std(recallscores))
print("AUROC Ave = " , np.mean(au_rocscores))

sm = SMOTE(random_state = 33)  
X_test_new, y_test_new = sm.fit_sample(X_test, y_test.ravel())

Y_pred_test = classifier.predict(X_test_new)
Y_pred_test_thres = (Y_pred_test > PREDICTION_THRESHOLD)
    #prob_y =classifier.predict_proba(reduced_features_scaled[test])
    #keep only positive class
    #prob_y=[p[0] for p in Y_pred_thres]
    #AUROC
au_roc_test = roc_auc_score(y_test_new,Y_pred_test)
    # Making the Confusion Matrix
cm_test = confusion_matrix(y_test_new, Y_pred_test_thres)
cr_test = classification_report(y_test_new,Y_pred_test_thres)

recall_test = cm_test[1,1] / (cm_test[1,1] + cm_test[1,0]) * 100
acc_test = (cm_test[0,0] + cm_test[1,1])/ (cm_test[0,0] + cm_test[1,1] + cm_test[0,1] + cm_test[1,0]) *100