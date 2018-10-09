#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:53:57 2018

@author: apple
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import catboost
import time
import os
import shutil
from matplotlib import pyplot as plt 

data = pd.read_csv('HUMANA_Data_cleaned.csv')
y_data = data['AMI_FLAG']
print(y_data)

positive_X = data[data['AMI_FLAG'] == 1]
positive_y = positive_X['AMI_FLAG']
negative_X = data[data['AMI_FLAG'] == 0]
negative_y = negative_X['AMI_FLAG']

positive_X = positive_X.drop(['AMI_FLAG'],axis=1)
negative_X = negative_X.drop(['AMI_FLAG'],axis=1)
positive_X.fillna(-999,inplace = True)
negative_X.fillna(-999,inplace = True)

ratio = 0.7
#X_train,X_test,y_train,y_test=train_test_split(xData,yData,train_size=ratio)
positive_X_train,positive_X_test,positive_y_train,positive_y_test=train_test_split(positive_X,positive_y,train_size=ratio)
negative_X_train,negative_X_test,negative_y_train,negative_y_test=train_test_split(negative_X,negative_y,train_size=ratio)
# positive_X_train = positive_X.sample(frac=0.7)
# positive_y_train = positive_X.sample(frac=0.7)
# positive_X_test = positive_X - positive_X_train
# positive_y_test = positive_y - positive_y_train

# negative_X_train = negative_X.sample(frac=0.7)
# negative_y_train = negative_X.sample(frac=0.7)
# negative_X_test = negative_X - negative_X_train
# negative_y_test = negative_y - negative_y_train

positive_train_len = len(positive_X_train)
negative_train_len = len(negative_X_train)
positive_test_len = len(positive_X_test)
negative_test_len = len(negative_X_test)
print('positive train num:', positive_train_len)
print('negative train num:', negative_train_len)
print('positive test num:', positive_test_len)
print('negative test num:', negative_test_len)

X_test = pd.concat([negative_X_test, positive_X_test])
y_test = pd.concat([negative_y_test, positive_y_test])
print('merged test')

models = []

model_nums = int(negative_train_len / positive_train_len) 
train = False
iterations = 10
depth = 5
learning_rate=0.5
loss_function='CrossEntropy'
path = './iter_' + str(iterations) + '_depth_' + str(depth) + '_lr_' + str(learning_rate) + '_' + loss_function + '/'

if train:
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)
	start_time = time.time()
	for i in range(model_nums):
		if i == model_nums - 1:
			negative_X_batch = negative_X_train[i*positive_train_len:]
			negative_y_batch = negative_y_train[i*positive_train_len:]
		else:
			negative_X_batch = negative_X_train[i*positive_train_len:(i+1)*positive_train_len]
			negative_y_batch = negative_y_train[i*positive_train_len:(i+1)*positive_train_len]

		X_train = [negative_X_batch, positive_X_train]
		X_train = pd.concat(X_train)
		y_train = [negative_y_batch, positive_y_train]
		y_train = pd.concat(y_train)
		print('train model {}, X_len {}, y_len {}'.format(i+1, len(X_train), len(y_train)))

		categorical_features_indices=np.where(X_train.dtypes!=np.float)[0]
		model=catboost.CatBoostClassifier(iterations=10, depth=5, learning_rate=0.5, loss_function='CrossEntropy',
                              logging_level='Verbose')
		model.fit(X_train, y_train,cat_features=categorical_features_indices)
		models.append(model)
		joblib.dump(model, path+'model_{}.joblib'.format(i+1)) 

votes = []
votes_prob = []
assert(os.path.exists(path))
for i in range(model_nums):
	if train:
		vote = np.array(models[i].predict(X_test))
		vote_prob = np.array(models[i].predict_proba(X_test)[:,1])
		# vote = (vote > 0.5).astype(int)
	else:
		tmp = joblib.load(path + 'model_{}.joblib'.format(i+1)) 
		print('load: ' + path + 'model_{}.joblib'.format(i+1))
		# import ipdb
		# ipdb.set_trace()
		vote = np.array(tmp.predict(X_test))
		vote_prob = np.array(tmp.predict_proba(X_test)[:,1])
		# vote = (vote > 0.5).astype(int)
	votes.append(vote)
	votes_prob.append(vote_prob)

votes = np.array(votes)
votes_prob = np.array(votes_prob)
res = np.sum(votes,0)
res = (res > model_nums/2).astype(int)

res_prob = np.mean(votes_prob, 0)
bins = np.linspace(0.0, 1.0, num=20)
plt.hist(res_prob, bins = bins) 
plt.title("prob histogram") 
plt.savefig(path+"prob_hist.png")
acc = np.sum(res == y_test)/len(y_test)
print('acc:', acc)
f = open(path+'acc.txt','w')
f.write(str(acc))
f.close()
print('1 num: ', np.sum(res))
