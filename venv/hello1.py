# 算法测试
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('C:/Users/hzp/Desktop/housePrice/input/train.csv')
test_df = pd.read_csv('C:/Users/hzp/Desktop/housePrice/input/test.csv')
train_data = pd.DataFrame(np.array([i for i in range(1,1461)]),columns = ['id'])
train_data.insert(0, 'MSSubClass' , train_df['MSSubClass'])
train_data.insert(1, 'LotArea', train_df['LotArea'])
train_data.insert(2, 'LotFrontage', train_df['LotFrontage'])

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

dt = DecisionTreeRegressor()
dt.fit(x_train , y_train)
y_pred = dt.predict(x_test)
acc_dt = round(dt.score(x_train , y_train) * 100 , 2)
print("acc_DecisionTree = ",acc_dt)

rf = RandomForestRegressor()
rf.fit(x_train , y_train)
y_pred = rf.predict(x_test)
acc_rf = round(rf.score(x_train , y_train) * 100 , 2)
print("acc_RandomForest = ",acc_rf)

xgb = XGBRegressor()
xgb.fit(x_train,y_train)
y_pred = xgb.predict(x_test)
acc_xgb = round(xgb.score(x_train , y_train) * 100 , 2)
print("acc_xgb = ",acc_xgb)

gbdt = GradientBoostingRegressor()
gbdt.fit(x_train , y_train)
y_pred = gbdt.predict(x_test)
acc_gbdt = round(gbdt.score(x_train , y_train) * 100 , 2)
print("acc_gbdt = ",acc_gbdt)


gbdt2 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1, min_samples_split=2, min_samples_leaf=1
        , max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False)
gbdt2.fit(x_train , y_train)
y_pred = gbdt2.predict(x_test)
acc_gbdt2 = round(gbdt2.score(x_train , y_train) * 100 , 2)
print("acc_gbdt2 = ",acc_gbdt2)

print(type(y_pred))
print(y_pred.shape)
submission = pd.DataFrame({
    "Id":test_data['id'],
    "SalePrice":y_pred
})
submission.to_csv('C:/Users/hzp/Desktop/housePrice/input/submission.csv',index=False)
