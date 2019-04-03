# 毕设用

#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('input_data/train.csv')

#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

#standardizing data
saleprice_scaled = StandardScaler().fit_transform(
    df_train['SalePrice'][:,np.newaxis]);

#deleting points
# print(df_train.sort_values(by = 'GrLivArea', ascending = False)[:2])
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

# print(df_train.info())
#convert categorical variable into dummy
df_train = pd.get_dummies(df_train)
print(df_train.shape)

xdata = df_train.drop('SalePrice', axis=1)
ydata = df_train['SalePrice']

print(xdata.shape)
print(ydata.shape)

xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.15)

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)
ytrue = np.array(ytest)
print("over")

def lin_model():
    lr_model = linear_model.LinearRegression()
    lr_model.fit(xtrain,ytrain)
    ypred = lr_model.predict(xtest)
    ss = (abs(ypred - ytrue)/ytrue).sum()/(ypred.shape[0])
    print("lr_model =", round((1-ss)*100, 2))
lin_model()

def svm_model():
    svm_model = svm.SVR()
    svm_model.fit(xtrain,ytrain)
    ypred = svm_model.predict(xtest)
    ss = (abs(ypred - ytrue)/ytrue).sum()/(ypred.shape[0])
    print("svm_model =", round((1-ss)*100, 2))
svm_model()

def dt_model():
    dt_model = DecisionTreeRegressor()
    dt_model.fit(xtrain,ytrain)
    ypred = dt_model.predict(xtest)
    ytrue = np.array(ytest)
    ss = (abs(ypred - ytrue)/ytrue).sum()/(ypred.shape[0])
    print("DecisionTree_model =", round((1-ss)*100, 2))
dt_model()

def rf_model():
    rf_model = RandomForestRegressor()
    rf_model.fit(xtrain,ytrain)
    ypred = rf_model.predict(xtest)
    ytrue = np.array(ytest)
    ss = (abs(ypred - ytrue)/ytrue).sum()/(ypred.shape[0])
    print("RandomForest_model =", round((1-ss)*100, 2))
rf_model()

def xgb_model():
    xgb_model = XGBRegressor()
    xgb_model.fit(xtrain,ytrain)
    ypred = xgb_model.predict(xtest)
    ytrue = np.array(ytest)
    ss = (abs(ypred - ytrue)/ytrue).sum()/(ypred.shape[0])
    print("xgb_model =", round((1-ss)*100, 2))
xgb_model()
