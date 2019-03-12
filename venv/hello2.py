# 数据预处理

import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('C:/Users/hzp/Desktop/housePrice/input/train.csv')
test_df = pd.read_csv('C:/Users/hzp/Desktop/housePrice/input/test.csv')

print(train_df.info())
print(train_df.shape)
train_data = pd.DataFrame(np.array([i for i in range(1,1461)]),columns = ['id'])
# print("train_data.info() = ", train_data.info())
train_data.insert(0, 'MSSubClass' , train_df['MSSubClass'])
train_data.insert(1, 'LotArea', train_df['LotArea'])
train_data.insert(2, 'LotFrontage', train_df['LotFrontage'])

# train_data['OverallQual'] = train_df['OverallQual']
# train_data['OverallCond'] = train_df['OverallCond']
# train_data['YearBuilt'] = train_df['YearBuilt']
# train_data['YearRemodAdd'] = train_df['YearRemodAdd']
# train_data['MasVnrArea'] = train_df['MasVnrArea']
# train_data['BsmtFinSF1'] = train_df['BsmtFinSF1']
# train_data['BsmtFinSF2'] = train_df['BsmtFinSF2']
# train_data['BsmtUnfSF'] = train_df['BsmtUnfSF']
# train_data['TotalBsmtSF'] = train_df['TotalBsmtSF']


x_train = train_data.drop('id',axis=1)
y_train = train_df['SalePrice']

test_data = pd.DataFrame(np.array([i for i in range(1461,2920)]),columns = ['id'])
test_data.insert(0, 'MSSubClass', test_df['MSSubClass'])
test_data.insert(1, 'LotArea', test_df['LotArea'])
test_data.insert(2,'LotFrontage',test_df['LotFrontage'])
x_test = test_data.drop('id',axis=1)

x_train = x_train.fillna(value=0)
x_test = x_test.fillna(value=0)
y_train = y_train.fillna(value=0)

logistic = LogisticRegression()
logistic.fit(x_train,y_train)
y_pred = logistic.predict(x_test)
acc_logistic = round(logistic.score(x_train , y_train) * 100 , 2)
print("acc_logistic = ",acc_logistic)

submission = pd.DataFrame({
    "Id":test_data['id'],
    "SalePrice":y_pred
})
submission.to_csv('C:/Users/hzp/Desktop/housePrice/input/submission.csv',index=False)