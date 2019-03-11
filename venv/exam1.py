import numpy as np
import pandas as pd

s = pd.Series([1,3,6,np.nan,44,1])
# print(s)

dates = pd.date_range('2019-02-27',periods=6)
# print(type(dates))
# print(dates)

df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
print(df)
# print(df['b'])

df1 = pd.DataFrame(np.arange(24).reshape(4,6))
# print(df1)
# print(df1.T)
# print(df1.sort_index(axis=1,ascending=True))
# print(df1.sort_index(axis=1,ascending=False))
# print(df1.sort_index(axis=0,ascending=False))
# print(df1.sort_index(axis=0,ascending=True))
# print(df1.loc[[1,3],2:4])
# print(df[df.c>=1])

