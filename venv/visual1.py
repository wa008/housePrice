# kaggle_kernels_first
#invite people for the Kaggle party

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv("input_data/train.csv")
# print(df_train.info())
# print(df_train.columns)
# sns.distplot(df_train['SalePrice']) # 将SalePrice可视化
# plt.title("SalePrice_probability_figure")
# plt.savefig("pictures/SalePrice_probability_figure.png")

# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
# data.plot.scatter(x=var, y='SalePrice')
# plt.title(var + '_SalePrice_scatter_plot') # 线性相关
# plt.savefig('pictures/' + var + '_SalePrice_scatter_plot.png') # 线性相关
# plt.show()


# var = 'YearBuilt'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
# f,ax = plt.subplots(figsize=(16,8))
# fig = sns.boxplot(x=var,y="SalePrice", data = data)
# # fig.axis(ymin=0,ymax=800000)
# plt.title(var + '_SalePrice_scatter_plot')
# plt.savefig('pictures/' + var + '_SalePrice_scatter_plot.png') # 线性相关
# plt.show()

# corrmat = df_train.corr()
# # f, ax = plt.subplots(figsize = (12,10))
# sns.heatmap(corrmat, vmax = .8,square = True) # 相关矩阵和热力图
# plt.title("data heatmap picture")
# plt.xticks(rotation=90)
# plt.yticks(rotation=360)
# plt.show()

# k = 10
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(df_train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True,fmt='.2f',annot_kws={'size':10},
#                  yticklabels=cols.values, xticklabels=cols.values)
# plt.title("zommed data heatmap picture")
# plt.xticks(rotation=90)
# plt.yticks(rotation=360)
# plt.show()

# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars',
#         'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df_train[cols], size = 2.5)
# plt.savefig('pictures/zoomed_correlation_figure.png') # 线性相关
# plt.show()

# 4.缺失值处理
# total = df_train.isnull().sum().sort_values(ascending=False)
# percent = (df_train.isnull().sum()/1460)\
#     .sort_values(ascending=False)
# missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
# print(missing_data.head(20))
#
# df_train = df_train.drop((missing_data[missing_data['Total']>1]).index,1)
# df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
# print(df_train.isnull().sum().max())

# saleprice_scaled = StandardScaler().fit_transform(
#     df_train['SalePrice'][:, np.newaxis])
# low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
# high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
# print('outer range (low) of the distribution:')
# print(low_range)
# print('\nouter range (high) of the distribution:')
# print(high_range)



df_train.sort_values(by='GrLivArea',ascending=False)[:2]
df_train = df_train.drop(df_train[df_train['Id']==1299].index)
df_train = df_train.drop(df_train[df_train['Id']==524].index)

# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
# data.plot.scatter(x=var, y='SalePrice',ylim=(0,800000))

# plt.title("GrLivArea_SalePrice_scatter_figure_after_delete_two_data")
# plt.savefig("pictures/GrLivArea_SalePrice_scatter_figure_agter_two_data.png")
# plt.show()

# var = 'GrLivArea'
# sns.distplot(df_train[var])
# plt.title("GrLivArea_probability_figure")
# plt.savefig("pictures/GrLivArea_probability_figure.png")

# var = 'TotalBsmtSF'
# plt.scatter(df_train[df_train[var]>0][var],df_train[df_train[var]>0]['SalePrice'])
# # plt.scatter(df_train[var],df_train['SalePrice'])
# plt.title(var + '_SalePrice_scatter_figure_exceptZero')
# plt.savefig('pictures/' + var + '_SalePrice_scatter_figure_exceptZero.png')
# plt.show()

df_train['SalePrice'] = np.log(df_train['SalePrice'])
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

# var = 'SalePrice'
# sns.distplot(df_train[var])
# plt.title(var + "_loged_probability_figure")
# plt.savefig("pictures/" + var + "_loged_probability_figure.png")

# var = 'GrLivArea'
# sns.distplot(df_train[var])
# plt.title(var + "_loged_probability_figure")
# plt.savefig("pictures/" + var + "_loged_probability_figure.png")

# sns.distplot(df_train['SalePrice'],fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['SalePrice'], plot=plt)
# plt.show()


# var = 'TotalBsmtSF'
# sns.distplot(df_train[var])
# plt.title(var + "_probability_figure")
# plt.savefig("pictures/" + var + "_probability_figure.png")
# plt.show()

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']),index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0, 'HasBsmt'] = 1
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

# var = 'TotalBsmtSF'
# sns.distplot(df_train[var])
# plt.xlim((4,9))
# plt.title(var + "_loged_probability_exceptZero_figure")
# plt.savefig("pictures/" + var + "_loged_probability_exceptZero_figure.png")
# plt.show()

# sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'],fit = norm)
# fig = plt.figure()
# res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot = plt)

# var = 'GrLivArea'
# # plt.scatter(df_train[df_train[var]>0][var],df_train[df_train[var]>0]['SalePrice'])
# plt.scatter(df_train[var],df_train['SalePrice'])
# plt.title(var + '_SalePrice_scatter_figure_loged')
# plt.savefig('pictures/' + var + '_SalePrice_scatter_figure_loged.png')
# plt.show()

# print(df_train.info())
df_train = pd.get_dummies(df_train)