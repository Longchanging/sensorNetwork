# coding:utf-8
'''
@time:    Created on  2018-10-15 20:40:32
@author:  Lanqing
@Func:    本质上是利用dataframe 数据检索， 不要走偏了
'''
import pandas as pd, numpy as np
##### Get data
df = pd.read_csv('fformationGT.csv', skiprows=0)
##### LineID to Time ID
df['index'], df['GroupID'], new_df = df.index, np.NaN, pd.DataFrame()
##### LineID to Group ID
for i in range(len(df.iloc[:, 0])):
    values = list(df.iloc[i, :][:10])
    groupID = '1' if 1.0 in values else '2'  if 14.0 in values else '3' if 17.0 in  values else '4' if 2.0 in  values else '0'
    df['GroupID'].iloc[i] = groupID
##### User ID to Group ID
for i in range(18):
    new_df = pd.DataFrame()
    loc = np.where(df.iloc[:, :12].values == ((i + 1) * 1.0))[0]  
##### Finally Fetch data, save into file
    new_df['index'] = df.iloc[loc, :]['index']
    new_df['Time'] = df.iloc[loc, 0]
    new_df['GroupID'] = df.iloc[loc, :]['GroupID']
    new_df.to_csv('../data/tmp/' + str(i + 1) + '.csv')