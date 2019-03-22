import pandas as pd
import numpy as np
import tensorflow as tf

train_df = pd.read_csv('input_data/train.csv')
test_df = pd.read_csv('input_data/test.csv')

train_data = pd.DataFrame(np.array([i for i in range(1,1461)]),columns = ['id'])
train_data.insert(0, 'Id', train_df['Id'])
train_data.insert(1, 'MSSubClass', train_df['MSSubClass'])
train_data.insert(2, 'LotFrontage', train_df['LotFrontage'])
train_data.insert(3, 'LotArea', train_df['LotArea'])
train_data.insert(4, 'OverallQual', train_df['OverallQual'])
train_data.insert(5, 'OverallCond', train_df['OverallCond'])
train_data.insert(6, 'YearBuilt', train_df['YearBuilt'])
train_data.insert(7, 'YearRemodAdd', train_df['YearRemodAdd'])
train_data.insert(8, 'MasVnrArea', train_df['MasVnrArea'])
train_data.insert(9, 'BsmtFinSF1', train_df['BsmtFinSF1'])
train_data.insert(10, 'BsmtFinSF2', train_df['BsmtFinSF2'])
train_data.insert(11, 'BsmtUnfSF', train_df['BsmtUnfSF'])
train_data.insert(12, 'TotalBsmtSF', train_df['TotalBsmtSF'])
train_data.insert(13, '1stFlrSF', train_df['1stFlrSF'])
train_data.insert(14, '2ndFlrSF', train_df['2ndFlrSF'])
train_data.insert(15, 'LowQualFinSF', train_df['LowQualFinSF'])
train_data.insert(16, 'GrLivArea', train_df['GrLivArea'])
train_data.insert(17, 'BsmtFullBath', train_df['BsmtFullBath'])
train_data.insert(18, 'BsmtHalfBath', train_df['BsmtHalfBath'])
train_data.insert(19, 'FullBath', train_df['FullBath'])
train_data.insert(20, 'HalfBath', train_df['HalfBath'])
train_data.insert(21, 'BedroomAbvGr', train_df['BedroomAbvGr'])
train_data.insert(22, 'KitchenAbvGr', train_df['KitchenAbvGr'])
train_data.insert(23, 'TotRmsAbvGrd', train_df['TotRmsAbvGrd'])
train_data.insert(24, 'Fireplaces', train_df['Fireplaces'])
train_data.insert(25, 'GarageYrBlt', train_df['GarageYrBlt'])
train_data.insert(26, 'GarageCars', train_df['GarageCars'])
train_data.insert(27, 'GarageArea', train_df['GarageArea'])
train_data.insert(28, 'WoodDeckSF', train_df['WoodDeckSF'])
train_data.insert(29, 'OpenPorchSF', train_df['OpenPorchSF'])
train_data.insert(30, 'EnclosedPorch', train_df['EnclosedPorch'])
train_data.insert(31, '3SsnPorch', train_df['3SsnPorch'])
train_data.insert(32, 'ScreenPorch', train_df['ScreenPorch'])
train_data.insert(33, 'PoolArea', train_df['PoolArea'])
train_data.insert(34, 'MiscVal', train_df['MiscVal'])
train_data.insert(35, 'MoSold', train_df['MoSold'])
train_data.insert(36, 'YrSold', train_df['YrSold'])
train_data.insert(37, 'SalePrice', train_df['SalePrice'])


test_data = pd.DataFrame(np.array([i for i in range(1461,2920)]),columns = ['id'])
test_data.insert(0, 'Id', test_df['Id'])
test_data.insert(1, 'MSSubClass', test_df['MSSubClass'])
test_data.insert(2, 'LotFrontage', test_df['LotFrontage'])
test_data.insert(3, 'LotArea', test_df['LotArea'])
test_data.insert(4, 'OverallQual', test_df['OverallQual'])
test_data.insert(5, 'OverallCond', test_df['OverallCond'])
test_data.insert(6, 'YearBuilt', test_df['YearBuilt'])
test_data.insert(7, 'YearRemodAdd', test_df['YearRemodAdd'])
test_data.insert(8, 'MasVnrArea', test_df['MasVnrArea'])
test_data.insert(9, 'BsmtFinSF1', test_df['BsmtFinSF1'])
test_data.insert(10, 'BsmtFinSF2', test_df['BsmtFinSF2'])
test_data.insert(11, 'BsmtUnfSF', test_df['BsmtUnfSF'])
test_data.insert(12, 'TotalBsmtSF', test_df['TotalBsmtSF'])
test_data.insert(13, '1stFlrSF', test_df['1stFlrSF'])
test_data.insert(14, '2ndFlrSF', test_df['2ndFlrSF'])
test_data.insert(15, 'LowQualFinSF', test_df['LowQualFinSF'])
test_data.insert(16, 'GrLivArea', test_df['GrLivArea'])
test_data.insert(17, 'BsmtFullBath', test_df['BsmtFullBath'])
test_data.insert(18, 'BsmtHalfBath', test_df['BsmtHalfBath'])
test_data.insert(19, 'FullBath', test_df['FullBath'])
test_data.insert(20, 'HalfBath', test_df['HalfBath'])
test_data.insert(21, 'BedroomAbvGr', test_df['BedroomAbvGr'])
test_data.insert(22, 'KitchenAbvGr', test_df['KitchenAbvGr'])
test_data.insert(23, 'TotRmsAbvGrd', test_df['TotRmsAbvGrd'])
test_data.insert(24, 'Fireplaces', test_df['Fireplaces'])
test_data.insert(25, 'GarageYrBlt', test_df['GarageYrBlt'])
test_data.insert(26, 'GarageCars', test_df['GarageCars'])
test_data.insert(27, 'GarageArea', test_df['GarageArea'])
test_data.insert(28, 'WoodDeckSF', test_df['WoodDeckSF'])
test_data.insert(29, 'OpenPorchSF', test_df['OpenPorchSF'])
test_data.insert(30, 'EnclosedPorch', test_df['EnclosedPorch'])
test_data.insert(31, '3SsnPorch', test_df['3SsnPorch'])
test_data.insert(32, 'ScreenPorch', test_df['ScreenPorch'])
test_data.insert(33, 'PoolArea', test_df['PoolArea'])
test_data.insert(34, 'MiscVal', test_df['MiscVal'])
test_data.insert(35, 'MoSold', test_df['MoSold'])
test_data.insert(36, 'YrSold', test_df['YrSold'])

y_train = train_data['SalePrice']
train_data = train_data.drop('Id', axis=1)
train_data = train_data.drop('SalePrice', axis=1)
x_test = test_data.drop('Id', axis=1)
x_train = train_data

x_train = x_train.fillna(value=0)
x_test = x_test.fillna(value=0)
y_train = y_train.fillna(value=0)
x_train = x_train.values
y_train = y_train.values
x_test = x_test.values

x_train = x_train.T
x_test = x_test.T
y_train = y_train.reshape(1460, 1)

n, m = x_train.shape
print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)

x_place = tf.placeholder(tf.float32, [37, None], name = "x_placeholder")
y_place = tf.placeholder(tf.float32, [None, 1], name = "y_placeholder")

layer_dimension = [37, 100, 300, 40, 1]
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

batch_size = 100
steps = 20000
dataset_size = x_train.shape[1]
print("dataset_size = ", dataset_size)
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    # print("before w[1] =", sess.run(w[2][2]))
    for i in range(steps):
        start = i * batch_size % dataset_size
        end = min(dataset_size, start + batch_size)
        sess.run(train_step, feed_dict = {x_place:x_train[:, start:end], y_place:y_train[start:end, :]})
        if i % 1000 == 0:
            loss_now_step = sess.run(cross_entropy, feed_dict = {x_place:x_train, y_place:y_train})
            print(i,loss_now_step)
    # print("after w[1] =", sess.run(w[2][2]))
    y_pred = sess.run(y, feed_dict={x_place:x_test})
    y_pred = y_pred.reshape(-1)
    print(y_pred)
    print(type(y_pred))
    print(y_pred.shape)

    submission = pd.DataFrame({
        "Id":test_data['id'],
        "SalePrice":y_pred
    })
    submission.to_csv('input_data/submission.csv',index=False)
