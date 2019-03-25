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

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
data.plot.scatter(x = var, y = 'SalePrice')
plt.savefig('pictures/TotalBsmtSF_SalePrice.png') # 线性相关
plt.show()
