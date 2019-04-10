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
import tensorflow as tf
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
# print(df_train.shape)
# print(df_train.info())

xdata = df_train.drop('SalePrice', axis=1)
ydata = df_train['SalePrice']

# print(xdata.shape)
# print(ydata.shape)

xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.15)
# print('xtrain.shape = ', xtrain.shape)
# print('xtest.shape =', xtest.shape)
# print('ytrain.shape =', ytrain.shape)
# print( 'ytest =', ytest.shape)
xtrain = np.array(xtrain)
xtest = np.array(xtest)
ytrain = np.array(ytrain)
ytest = np.array(ytest)
ytrue = ytest
# print(type(xtrain))
print("data Characteristic engineering over")

def neural_netword():
    global xtrain
    global xtest
    global ytrain
    global ytest
    global ytrue
    print("neural_network begin")
    # print(type(xtrain))
    x_train = xtrain.T
    y_train = ytrain.T.reshape((-1,1))
    x_test = xtest.T
    y_test = ytest.T.reshape((-1,1))
    n, m = x_train.shape
    # print("x_train.shape = ", x_train.shape)
    # print("y_train.shape = ", y_train.shape)

    x_place = tf.placeholder(tf.float32, [n, None], name = "x_placeholder")
    y_place = tf.placeholder(tf.float32, [None, 1], name = "y_placeholder")

    layer_dimension = [n, 400, 500, 100, 200, 50, 1]
    n_layers = len(layer_dimension)
    w = [0 for i in range(n_layers)]
    b = [0 for i in range(n_layers)]
    a = [0 for i in range(n_layers)]
    for i in range(1, n_layers):
        w[i] = tf.Variable(tf.random_normal([layer_dimension[i], layer_dimension[i-1]], stddev = 1, dtype = tf.float32))
        b[i] = tf.Variable(tf.random_normal([layer_dimension[i], 1], stddev = 1, dtype = tf.float32))

    a[0] = x_place
    y = 0
    for i in range(1, n_layers):
        if i == n_layers - 1:
            y = tf.matmul(w[i], a[i-1]) + b[i]
        else:
            a[i] = tf.nn.relu(tf.matmul(w[i], a[i-1]) + b[i])

    cross_entropy = tf.reduce_mean(tf.square(y_place - y))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    batch_size = 1500
    steps = 200000
    dataset_size = xtrain.shape[1]
    # print("dataset_size = ", dataset_size)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(steps):
            start = i * batch_size % dataset_size
            end = min(dataset_size, start + batch_size)
            sess.run(train_step, feed_dict = {x_place:x_train[:, start:end], y_place:y_train[start:end, :]})
            if i % 1000 == 0:
                pass
                # loss_now_step = sess.run(cross_entropy, feed_dict = {x_place:x_train, y_place:y_train})
                # print(i,loss_now_step)
        ypred = sess.run(y, feed_dict = {x_place:x_test})
        ypred = ypred.reshape(-1)
        ss = (ypred - ytrue).dot(ypred - ytrue)/(ypred.shape[0])
        print("neural_network_square_loss =", ss)
        ss = (abs(ypred - ytrue)/ytrue).sum()/(ypred.shape[0])
        # print('ss = ', ss)
        print("neural_netword_model =", round((1-ss)*100, 2))

def lin_model():
    lr_model = linear_model.LinearRegression()
    lr_model.fit(xtrain,ytrain)
    ypred = lr_model.predict(xtest)
    ss = (ypred - ytrue).dot(ypred - ytrue)/(ypred.shape[0])
    print("\nlr_model_square loss =", ss)
    ss = (abs(ypred - ytrue)/ytrue).sum()/(ypred.shape[0])
    print("lr_model =", round((1-ss)*100, 2))
lin_model()

def svm_model():
    svm_model = svm.SVR()
    svm_model.fit(xtrain,ytrain)
    ypred = svm_model.predict(xtest)
    ss = (ypred - ytrue).dot(ypred - ytrue)/(ypred.shape[0])
    print("\nsvm_model_square loss =", ss)
    ss = (abs(ypred - ytrue)/ytrue).sum()/(ypred.shape[0])
    print("svm_model =", round((1-ss)*100, 2))
svm_model()

def dt_model():
    dt_model = DecisionTreeRegressor()
    dt_model.fit(xtrain,ytrain)
    ypred = dt_model.predict(xtest)
    ytrue = np.array(ytest)
    ss = (ypred - ytrue).dot(ypred - ytrue)/(ypred.shape[0])
    print("\ndt_model_square loss =", ss)
    ss = (abs(ypred - ytrue)/ytrue).sum()/(ypred.shape[0])
    print("DecisionTree_model =", round((1-ss)*100, 2))
dt_model()

def rf_model():
    rf_model = RandomForestRegressor()
    rf_model.fit(xtrain,ytrain)
    ypred = rf_model.predict(xtest)
    ytrue = np.array(ytest)
    ss = (ypred - ytrue).dot(ypred - ytrue)/(ypred.shape[0])
    print("\nrf_model_square loss =", ss)
    ss = (abs(ypred - ytrue)/ytrue).sum()/(ypred.shape[0])
    print("RandomForest_model =", round((1-ss)*100, 2))
rf_model()

def xgb_model():
    xgb_model = XGBRegressor()
    xgb_model.fit(xtrain,ytrain)
    ypred = xgb_model.predict(xtest)
    ytrue = np.array(ytest)
    ss = (ypred - ytrue).dot(ypred - ytrue)/(ypred.shape[0])
    print("\nxgb_model_square loss =", ss)
    ss = (abs(ypred - ytrue)/ytrue).sum()/(ypred.shape[0])
    # print('ss =', ss)
    print("xgb_model =", round((1-ss)*100, 2))
xgb_model()


neural_netword()
