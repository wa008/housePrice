# 毕设用

# invite people for the Kaggle party
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import tensorflow as tf
import time
import warnings

warnings.filterwarnings('ignore')

df_train = pd.read_csv('input_data/train.csv')

# missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

# standardizing data
saleprice_scaled = StandardScaler().fit_transform(
    df_train['SalePrice'][:, np.newaxis]);

# deleting points
# print(df_train.sort_values(by = 'GrLivArea', ascending = False)[:2])
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

# applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
# data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
# create column for new variable (one is enough because it's a binary categorical feature)
# if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
# transform data
df_train.loc[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

# print(df_train.info())
# convert categorical variable into dummy
df_train = pd.get_dummies(df_train)
# print(df_train.shape)
# print(df_train.info())

xdata = df_train.drop('SalePrice', axis=1)
ydata = df_train['SalePrice']

# print(xdata.shape)
# print(ydata.shape)

df_train = StandardScaler().fit(df_train)
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
print("ytrain.shape =", ytrain.shape)
print("data Characteristic engineering over")


def lin_model():
    lr_model = linear_model.LinearRegression(normalize=True)  # 数据归一化
    lr_model.fit(xtrain, ytrain)
    ypred = lr_model.predict(xtest)
    ss = (ypred - ytrue).dot(ypred - ytrue) / (ypred.shape[0])
    print("\nlr_model_square loss =", ss)
    ss = (abs(ypred - ytrue) / ytrue).sum() / (ypred.shape[0])
    print("lr_model =", round((1 - ss) * 100, 2))


# lin_model()


def svm_model_update_parameter_before():
    svm_model = svm.SVR()
    svm_model.fit(xtrain, ytrain)
    ypred = svm_model.predict(xtest)
    ss = (ypred - ytrue).dot(ypred - ytrue) / (ypred.shape[0])
    print("\nsvm_model_update_parameter_before square loss =", ss)
    ss = (abs(ypred - ytrue) / ytrue).sum() / (ypred.shape[0])
    print("svm_model_update_parameter_before accuracy =", round((1 - ss) * 100, 2))


def svm_model_update_parameter_after():
    kernel_list = ['rbf', 'sigmoid']
    # kernel_list = ['callable']
    tol_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    # tol_list = [1e-3]
    # print("tol_list = ", tol_list)
    score_max = -1e20
    kernel_fin = ""
    tol_fin = 0
    for kernel_step in kernel_list:
        for tol_step in tol_list:
            score_now = 0.0
            k = 10
            for i in range(k):  # k折交叉验证
                n, m = xtrain.shape
                svm_model = svm.SVR(kernel=kernel_step, tol=tol_step)
                xtrain_fit = np.concatenate((xtrain[:n // k * i, :], xtrain[min(n, n // k * (i + 1)):n, :]), axis=0)
                ytrain_fit = np.concatenate((ytrain[:n // k * i], ytrain[min(n, n // k * (i + 1), n):]))
                xtest_score = xtrain[n // k * i:min(n, n // k * (i + 1)), :]
                ytest_score = ytrain[n // k * i:min(n, n // k * (i + 1))]
                # print("\nshape,i,m,n= ",i,m,n)
                # print(xtrain_fit.shape)
                # print(ytrain_fit.shape)
                # print(xtest_score.shape)
                # print(ytest_score.shape)
                svm_model.fit(xtrain_fit, ytrain_fit)
                score_now_now = svm_model.score(xtest_score, ytest_score) / xtest_score.shape[0]
                score_now += score_now_now
            if score_now > score_max:
                score_max = score_now
                kernel_fin = kernel_step
                tol_fin = tol_step
            # print("kernel_step, tol_step, now score = ", kernel_step, tol_step, score_now)
    # print("best parameter = ", kernel_fin, tol_fin)
    svm_model = svm.SVR(kernel=kernel_fin, tol=tol_fin)
    svm_model.fit(xtrain, ytrain)
    ypred = svm_model.predict(xtest)
    ss = (ypred - ytrue).dot(ypred - ytrue) / (ypred.shape[0])
    print("\nsvm_model_update_parameter_after_square loss =", ss)
    ss = (abs(ypred - ytrue) / ytrue).sum() / (ypred.shape[0])
    print("svm_model_update_parameter_after accuracy =", round((1 - ss) * 100, 2))


def svm_model():
    svm_model_update_parameter_before()
    svm_model_update_parameter_after()


# svm_model()


def dt_model_update_parameter_before():
    dt_model = DecisionTreeRegressor()
    dt_model.fit(xtrain, ytrain)
    ypred = dt_model.predict(xtest)
    ytrue = np.array(ytest)
    ss = (ypred - ytrue).dot(ypred - ytrue) / (ypred.shape[0])
    print("\ndt_model_update_parameter_before square loss =", ss)
    ss = (abs(ypred - ytrue) / ytrue).sum() / (ypred.shape[0])
    print("DecisionTree_model_update_parameter_before =", round((1 - ss) * 100, 2))


def dt_model_update_parameter_after():
    min_samples_split_fin = 0
    min_samples_leaf_fin = 0
    min_impurity_split_fin = 0
    score_fin = -1e20
    min_impurity_split_list = [1e-4, ]
    for i in range(1, 10):
        min_impurity_split_list.append(min_impurity_split_list[len(min_impurity_split_list) - 1] * 0.1)
    # print("min_impurity_split_list = ", min_impurity_split_list)
    for min_samples_split_now in range(2, 8):
        for min_samples_leaf_now in range(1, min_samples_split_now):
            for min_impurity_split_now in min_impurity_split_list:
                score_now = 0.0
                k = 10
                for i in range(k):  # k折交叉验证
                    n, m = xtrain.shape
                    dt_model = DecisionTreeRegressor(min_samples_split=min_samples_split_now,
                                                     min_samples_leaf=min_samples_leaf_now,
                                                     min_impurity_split=min_impurity_split_now)
                    xtrain_fit = np.concatenate((xtrain[:n // k * i, :], xtrain[min(n, n // k * (i + 1)):n, :]), axis=0)
                    ytrain_fit = np.concatenate((ytrain[:n // k * i], ytrain[min(n, n // k * (i + 1), n):]))
                    xtest_score = xtrain[n // k * i:min(n, n // k * (i + 1)), :]
                    ytest_score = ytrain[n // k * i:min(n, n // k * (i + 1))]
                    # print("\nshape,i,m,n= ",i,m,n)
                    # print(xtrain_fit.shape)
                    # print(ytrain_fit.shape)
                    # print(xtest_score.shape)
                    # print(ytest_score.shape)
                    dt_model.fit(xtrain_fit, ytrain_fit)
                    score_now_now = dt_model.score(xtest_score, ytest_score) / xtest_score.shape[0]
                    score_now += score_now_now
                if score_now > score_fin:
                    score_fin = score_now
                    min_samples_split_fin = min_samples_split_now
                    min_samples_leaf_fin = min_samples_leaf_now
                    min_impurity_split_fin = min_impurity_split_now
                # print("min_samples_split_now, score_now = ", min_samples_split_now, score_now)
    # print("best parameter min_samples_split, min_samples_leaf, min_impurity_split =",
    #       min_samples_split_fin, min_samples_leaf_fin, min_impurity_split_fin)
    dt_model = DecisionTreeRegressor(min_samples_split=min_samples_split_fin,
                                     min_samples_leaf=min_samples_leaf_fin, min_impurity_split=min_impurity_split_fin)
    dt_model.fit(xtrain, ytrain)
    ypred = dt_model.predict(xtest)
    ytrue = np.array(ytest)
    ss = (ypred - ytrue).dot(ypred - ytrue) / (ypred.shape[0])
    print("\ndt_model_update_parameter_after square loss =", ss)
    ss = (abs(ypred - ytrue) / ytrue).sum() / (ypred.shape[0])
    print("DecisionTree_model_update_parameter_after =", round((1 - ss) * 100, 2))


def dt_model():
    dt_model_update_parameter_before()
    dt_model_update_parameter_after()


# dt_model()


def rf_model_updata_parameter_before():
    rf_model = RandomForestRegressor()
    rf_model.fit(xtrain, ytrain)
    ypred = rf_model.predict(xtest)
    ytrue = np.array(ytest)
    ss = (ypred - ytrue).dot(ypred - ytrue) / (ypred.shape[0])
    print("RandomForest_model_updata_parameter_before square_loss =", ss)
    ss = (abs(ypred - ytrue) / ytrue).sum() / (ypred.shape[0])
    print("RandomForest_model_updata_parameter_before accuracy =", round((1 - ss) * 100, 2))


def rf_model_update_parameter_after():
    n_estimators_list = [i * 20 for i in range(1, 13, 2)]
    # min_samples_split_list = []
    # min_samples_leaf_list = []
    max_features_list = ['auto', 'log2', 'sqrt', 60, 100, 160, 200]
    # n_estimators_list = [50]
    # max_features_list = ['auto']
    n_estimators_fin = 0
    min_samples_leaf_fin = 0
    min_samples_split_fin = 0
    max_features_fin = 0
    score_fin = -1e20
    for n_estimators_now in n_estimators_list:
        for min_samples_split_now in range(2, 7):
            for min_samples_leaf_now in range(1, min_samples_split_now):
                for max_features_now in max_features_list:
                    score_now = 0.0
                    k = 10
                    for i in range(k):  # k折交叉验证
                        n, m = xtrain.shape
                        xtrain_fit = np.concatenate((xtrain[:n // k * i, :], xtrain[min(n, n // k * (i + 1)):n, :]), axis=0)
                        ytrain_fit = np.concatenate((ytrain[:n // k * i], ytrain[min(n, n // k * (i + 1), n):]))
                        xtest_score = xtrain[n // k * i:min(n, n // k * (i + 1)), :]
                        ytest_score = ytrain[n // k * i:min(n, n // k * (i + 1))]
                        # print("\nshape,i,m,n= ",i,m,n)
                        # print(xtrain_fit.shape)
                        # print(ytrain_fit.shape)
                        # print(xtest_score.shape)
                        # print(ytest_score.shape)e
                        rf_model = RandomForestRegressor(n_estimators=n_estimators_now, min_samples_split=min_samples_split_now,
                                                         min_samples_leaf=min_samples_leaf_now, max_features=max_features_now)
                        rf_model.fit(xtrain_fit, ytrain_fit)
                        score_now_now = rf_model.score(xtest_score, ytest_score) / xtest_score.shape[0]
                        score_now += score_now_now
                    if score_now > score_fin:
                        score_fin = score_now
                        n_estimators_fin = n_estimators_now
                        min_samples_split_fin = min_samples_split_now
                        min_samples_leaf_fin = min_samples_leaf_now
                        max_features_fin = max_features_now
        # print("score_now = ", score_now)
    # print("best parameter n_estimators_fin, min_samples_split_fin, min_samples_leaf_fin, max_features_fin =",
    #       n_estimators_fin, min_samples_split_fin, min_samples_leaf_fin, max_features_fin)
    rf_model = RandomForestRegressor(n_estimators=n_estimators_fin, min_samples_split=min_samples_split_fin,
                                    min_samples_leaf=min_samples_leaf_fin, max_features=max_features_fin)
    rf_model.fit(xtrain, ytrain)
    ypred = rf_model.predict(xtest)
    ytrue = np.array(ytest)
    ss = (ypred - ytrue).dot(ypred - ytrue) / (ypred.shape[0])
    print("randomForest_model_update_parameter_after square_loss =", ss)
    ss = (abs(ypred - ytrue) / ytrue).sum() / (ypred.shape[0])
    print("RandomForest_model_update_parameter_after accuracy=", round((1 - ss) * 100, 2))

def rf_model():
    print("")
    rf_model_updata_parameter_before()
    rf_model_update_parameter_after()
# rf_model()

def xgb_model_update_parameter_before():
    xgb_model = XGBRegressor()
    xgb_model.fit(xtrain, ytrain)
    ypred = xgb_model.predict(xtest)
    ytrue = np.array(ytest)
    ss = (ypred - ytrue).dot(ypred - ytrue) / (ypred.shape[0])
    print("xgb_model_update_parameter_before square_loss =", ss)
    ss = (abs(ypred - ytrue) / ytrue).sum() / (ypred.shape[0])
    # print('ss =', ss)
    print("xgb_model_update_parameter_before accuracy=", round((1 - ss) * 100, 2))

def xgb_model_update_parameter_after():
    # booster = ['gbtree', 'gbliner']
    learning_rate_list = [i*0.01 for i in range(1,34,3)]
    max_depth_list = list(range(1,8,3))
    gamma_list = [1]
    for i in range(4):
        gamma_list.append(gamma_list[len(gamma_list)-1]*0.1)
    gamma_list.append(0)
    lambda_list = [0.01, 0.1, 0.15, 1]
    alpha_list = [0.01, 0.1, 0.15, 1]
    min_child_weight_list = [0.1, 0.5, 1, 5]
    steps = len(learning_rate_list)*len(max_depth_list)*len(gamma_list)*len(lambda_list)*len(alpha_list)*len(min_child_weight_list)
    # print("steps = ", steps)
    # learning_rate_list = [0.2]
    # max_depth_list = [10]
    # gamma_list = [0]
    # lambda_list = [1]
    # alpha_list = [1]
    learning_rate_fin = 0
    max_depth_fin = 0
    gamma_fin = 0
    lambda_fin = 0
    alpha_fin = 0
    score_fin = -1e20
    time_prin = time.clock()
    def k_cv(learning_rate_now, max_depth_now, gamma_now, lambda_now, alpha_now, min_child_weight_now):
        score_now = 0
        k = 10
        for i in range(k):  # k折交叉验证
            n, m = xtrain.shape
            xtrain_fit = np.concatenate((xtrain[:n // k * i, :], xtrain[min(n, n // k * (i + 1)):n, :]), axis=0)
            ytrain_fit = np.concatenate((ytrain[:n // k * i], ytrain[min(n, n // k * (i + 1), n):]))
            xtest_score = xtrain[n // k * i:min(n, n // k * (i + 1)), :]
            ytest_score = ytrain[n // k * i:min(n, n // k * (i + 1))]
            # print("\nshape,i,m,n= ",i,m,n)
            # print(xtrain_fit.shape)
            # print(ytrain_fit.shape)
            # print(xtest_score.shape)
            # print(ytest_score.shape)
            xgb_model = XGBRegressor(learning_rate=learning_rate_now, max_depth=max_depth_now, gamma=gamma_now,reg_lambda=lambda_now,
                                     reg_alpha=alpha_now, min_child_weight=min_child_weight_now)
            xgb_model.fit(xtrain_fit, ytrain_fit)
            score_now_now = xgb_model.score(xtest_score, ytest_score) / xtest_score.shape[0]
            score_now += score_now_now
        return score_now
    for learning_rate_now in learning_rate_list:
        for max_depth_now in max_depth_list:
            for gamma_now in gamma_list:
                for lambda_now in lambda_list:
                    for alpha_now in alpha_list:
                        for min_child_weight_now in min_child_weight_list:
                            time_now = time.clock()
                            score_now = k_cv(learning_rate_now, max_depth_now, gamma_now, lambda_now, alpha_now, min_child_weight_now)
                            if score_now > score_fin:
                                score_fin = score_now
                                learning_rate_fin = learning_rate_now
                                max_depth_fin = max_depth_now
                                gamma_fin = gamma_now
                                lambda_fin = lambda_now
                                alpha_fin = alpha_now
                            if int(time.clock() - time_prin) > 60:
                                pass
                                print("one time = ", time.clock() - time_now)
                                # print("all time = ", (time.clock() - time_now)*steps)
                                print("all time = ", (time.clock() - time_now)*steps/60)
                                time_prin = time.clock()
        # print("learning_rate_now, max_depth_now, score_now = ",learning_rate_now, max_depth_now, score_now)
    # print('best parameter learning_rate_fin, max_depth_fin, gamma_fin, lambda_fin, alpha_fin = ',
    #       learning_rate_fin, max_depth_fin, gamma_fin, lambda_fin, alpha_fin)
    xgb_model = XGBRegressor(learning_rate=learning_rate_fin, max_depth=max_depth_fin, gamma=gamma_fin,reg_lambda=lambda_fin,reg_alpha=alpha_fin)
    xgb_model.fit(xtrain, ytrain)
    ypred = xgb_model.predict(xtest)
    ytrue = np.array(ytest)
    ss = (ypred - ytrue).dot(ypred - ytrue) / (ypred.shape[0])
    print("xgb_model_update_parameter_after square_loss = ", ss)
    ss = (abs(ypred - ytrue) / ytrue).sum() / (ypred.shape[0])
    # print('ss =', ss)
    print("xgb_model_update_parameter_after accuracy = ", round((1 - ss) * 100, 2))
# 最小结点权重
def xgb_model():
    ou = time.asctime(time.localtime(time.time()))
    print(ou)
    print("")
    xgb_model_update_parameter_before()
    xgb_model_update_parameter_after()
xgb_model()


def neural_netword():
    global xtrain
    global xtest
    global ytrain
    global ytest
    global ytrue
    # print("neural_network begin")
    # print(type(xtrain))
    x_train = xtrain.T
    y_train = ytrain.T.reshape((-1, 1))
    x_test = xtest.T
    y_test = ytest.T.reshape((-1, 1))
    n, m = x_train.shape
    # print("x_train.shape = ", x_train.shape)
    # print("y_train.shape = ", y_train.shape)

    x_place = tf.placeholder(tf.float32, [n, None], name="x_placeholder")
    y_place = tf.placeholder(tf.float32, [None, 1], name="y_placeholder")

    layer_dimension = [n, 400, 500, 100, 200, 50, 1]
    n_layers = len(layer_dimension)
    w = [0 for i in range(n_layers)]
    b = [0 for i in range(n_layers)]
    a = [0 for i in range(n_layers)]
    for i in range(1, n_layers):
        w[i] = tf.Variable(tf.random_normal([layer_dimension[i], layer_dimension[i - 1]], stddev=1, dtype=tf.float32))
        b[i] = tf.Variable(tf.random_normal([layer_dimension[i], 1], stddev=1, dtype=tf.float32))

    a[0] = x_place
    y = 0
    for i in range(1, n_layers):
        if i == n_layers - 1:
            y = tf.matmul(w[i], a[i - 1]) + b[i]
        else:
            a[i] = tf.nn.relu(tf.matmul(w[i], a[i - 1]) + b[i])

    cross_entropy = tf.reduce_mean(tf.square(y_place - y))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    batch_size = 1500
    steps = 20000
    dataset_size = xtrain.shape[1]
    # print("dataset_size = ", dataset_size)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(steps):
            start = i * batch_size % dataset_size
            end = min(dataset_size, start + batch_size)
            sess.run(train_step, feed_dict={x_place: x_train[:, start:end], y_place: y_train[start:end, :]})
            if i % 1000 == 0:
                pass
                loss_now_step = sess.run(cross_entropy, feed_dict={x_place: x_train, y_place: y_train})
                # print('steps, square_loss =', i, loss_now_step)
        ypred = sess.run(y, feed_dict={x_place: x_test})
        ypred = ypred.reshape(-1)
        ss = (ypred - ytrue).dot(ypred - ytrue) / (ypred.shape[0])
        print("neural_network_square_loss =", ss)
        ss = (abs(ypred - ytrue) / ytrue).sum() / (ypred.shape[0])
        # print('ss = ', ss)
        print("neural_netword_model =", round((1 - ss) * 100, 2))

neural_netword()