import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

train_data = pd.read_csv("input_data/train.csv")
test_data = pd.read_csv("input_data/test.csv")
print("train_data.shape =", train_data.shape)
print("test_data.shape =", test_data.shape)
# print(train_data.info())
train_data.drop(train_data[(train_data["GrLivArea"]>4000) & (train_data["SalePrice"
                ]<300000)].index, inplace = True)
# plt.figure(figsize = (15, 8))
# sns.boxplot(train_data.YearBuilt, train_data.SalePrice)

full_data = pd.concat([train_data, test_data], ignore_index=True)
full_data.drop(['Id'], axis = 1, inplace = True)
print("full_data.shape =", full_data.shape)

cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    full_data[col].fillna("None", inplace=True)
    # PoolQC表示游泳池数量，空值代表没有，这些缺失值不重要，用None代替

cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    full_data[col].fillna(0, inplace=True)
    # 这些缺失值为面积，没有代表为0，用0补充

# 中位数插补
full_data['LotFrontage']=full_data.groupby(['LotArea'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# aa = full_data.isnull().sum()
# print(aa[aa>0].sort_values(ascending = False))

print(full_data.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count']))

# plt.show()
