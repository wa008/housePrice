# kaggle_kernels_first
#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv("input_data/train.csv")
# print(df_train.columns)
# sns.distplot(df_train['SalePrice']) # 将SalePrice可视化

# var = 'YearBuilt'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
# f,ax = plt.subplots(figsize=(16,8))
# fig = sns.boxplot(x=var,y="SalePrice", data = data)
# fig.axis(ymin=0,ymax=800000)
# plt.savefig('pictures/SalePrice_YearBuilt.png') # 线性相关
# plt.show()

corrmat = df_train.corr()
# f,ax = plt.subplots(figsize = (12,9))
# sns.heatmap(corrmat, vmax = 1,square = True) # 相关矩阵和热力图

k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,fmt='.2f',annot_kws={'size':10},
                 yticklabels=cols.values, xticklabels=cols.values)
# plt.savefig('pictures/zoomed_correlation_matrix.png') # 线性相关

plt.show()
