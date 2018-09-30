import catboost as cb
import pandas as pd
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from utils import auc

from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('TAMU_FINAL_DATASET_2018.csv')
y_data = data['AMI_FLAG']
X_data = data.drop(['AMI_FLAG'],axis=1)


rows = X_data.shape[0]
columns = X_data.shape[1]

# print(data.dtypes)
feature_num = data.shape[1]
cat_features_index = []
for (i,d) in enumerate(X_data.dtypes):
	if d == 'object':
		X_data.iloc[:,i].fillna("-99999", inplace=True)
		cat_features_index.append(i)
		# X_data.iloc[:,i] = str(X_data.iloc[:,i])
	else:
		X_data.iloc[:,i].fillna(-99999, inplace=True)

print(cat_features_index)

# cat_features_index.remove(413)
# print(X_data.iloc[:,cat_features_index])

test_num = 5000


X_train = X_data[test_num:]
y_train = y_data[test_num:]
y_train = y_train.reshape((len(y_train),1))

X_test = X_data[:test_num]
y_test = y_data[:test_num]
y_test = y_test.reshape((len(y_test),1))

print(X_train.shape)
print(y_train.shape)

#With Categorical features
# clf = cb.CatBoostClassifier(eval_metric="AUC", depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15)
# clf.fit(train_data,y_train, cat_features_index)
# print("auc of clf: ", auc(clf, train_data, test_data))

#One hot encoding:
print('start encoding')
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_data)
enc_X_train = enc.transform(X_train).toarray()
enc_X_test = enc.transform(X_test)

params = {'depth': [4, 7, 10],
          'learning_rate' : [0.03, 0.1, 0.15],
         'l2_leaf_reg': [1,4,9],
         'iterations': [300]}

print('start catboost')
cb = cb.CatBoostClassifier()
cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv = 3)
cb_model.fit(enc_X_train, y_train)
print("auc of clf: ", auc(cb, enc_X_train, enc_X_test))


