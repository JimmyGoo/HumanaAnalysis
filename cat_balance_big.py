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
import matplotlib
matplotlib.use("tkagg")
from matplotlib import pyplot as plt 

def train(train=False,iterations=10,depth=10,learning_rate=None,loss_function='CrossEntropy',l2_leaf_reg=9):
	data = pd.read_csv('HUMANA_Data_cleaned.csv')
	y_data = data['AMI_FLAG']

	positive_X = data[data['AMI_FLAG'] == 1]
	positive_y = positive_X['AMI_FLAG']
	negative_X = data[data['AMI_FLAG'] == 0]
	negative_y = negative_X['AMI_FLAG']

	positive_X = positive_X.drop(['AMI_FLAG'],axis=1)
	negative_X = negative_X.drop(['AMI_FLAG'],axis=1)
	positive_X.fillna(-999,inplace = True)
	negative_X.fillna(-999,inplace = True)

	ratio = 0.85
	#X_train,X_test,y_train,y_test=train_test_split(xData,yData,train_size=ratio)
	positive_X_train,positive_X_test,positive_y_train,positive_y_test=train_test_split(positive_X,positive_y,train_size=ratio)
	negative_X_train,negative_X_test,negative_y_train,negative_y_test=train_test_split(negative_X,negative_y,train_size=ratio)

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

	models = []
	model_nums = int(negative_train_len / positive_train_len)

	path = './iter_' + str(iterations) + '_depth_' + str(depth) + '_lr_' + str(learning_rate) + '_' + loss_function + '_l2_' + str(l2_leaf_reg) + '/'

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
			model=catboost.CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate, loss_function=loss_function,
								  logging_level='Verbose', l2_leaf_reg=l2_leaf_reg)
			model.fit(X_train, y_train,cat_features=categorical_features_indices)
			models.append(model)
			joblib.dump(model, path+'model_{}.joblib'.format(i+1)) 

	votes = []
	votes_prob = []
	assert(os.path.exists(path))
	importances_dict = {}
	for i in range(model_nums):
		if train:
			current_model = models[i]
		else:
			current_model = joblib.load(path + 'model_{}.joblib'.format(i+1)) 

		vote = np.array(current_model.predict(X_test))
		vote_prob = np.array(current_model.predict_proba(X_test)[:,1])
		importances = np.argsort(current_model.get_feature_importance())[-10:]

		for idx in importances:
			importances_name = X_test.columns.values[idx]
			if not importances_name in importances_dict:
				importances_dict[importances_name] = 1
			else:
				importances_dict[importances_name] += 1

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
	f1 = metrics.f1_score(y_test, res)
	print('f1: ', f1)
	top20importance = sorted(importances_dict.items(), key=lambda x:x[1])[-20:]
	top20importance = top20importance[::-1]
	print('importances dict: ', top20importance)
	f = open(path+'info.txt','w')
	f.write('acc: '+str(acc))
	f.write('fscore: '+str(f1))
	f.write('')
	for i in top20importance:
		f.write(i[0] + ' ' + str(i[1]))
	f.close()

	print('1 num: ', np.sum(res))
	print('path:', path)

if __name__ == '__main__':
    
 #    iterations = [10,20,30,40,50] #best 10
	# depths = [6,7,8,9,10] # best 10
	# learning_rates=[None,0.1,0.2,0.3,0.4,0.5]
	# loss_functions=['Logloss','CrossEntropy']
	# l2_leaf_reg=9

	iterations = [50]
	depths = [10] # best 10
	learning_rates=[0.15]
	loss_functions=['CrossEntropy']
	l2_leaf_reg=9

	for i in iterations:
		for d in depths:
			for lr in learning_rates:
				for lf in loss_functions:
					train(train=True,iterations=i,depth=d,learning_rate=lr,loss_function=lf,l2_leaf_reg=9)
