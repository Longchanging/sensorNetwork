# coding:utf-8
'''
@time:    Created on  2018-10-17 11:14:47
@author:  Lanqing
@Func:    src.simple_version
'''
import pandas as pd, matplotlib.pyplot as plt, numpy as np, os
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

input_folder, label_folder = '../data/input/scene_present/accel_fast/', '../data/tmp/'

def data_prepare():
    info = []
    count = 0
    for i in range(18): 
        print(i)
        print(count)
        if os.path.exists(input_folder + str('%.2d' % (i + 1)) + '.csv'):
            man_file = pd.read_csv(input_folder + str('%.2d' % (i + 1)) + '.csv')  # read
            man_file.columns = ['Tid', 'X', 'Y', 'Z']
            label_file = pd.read_csv(label_folder + str(i + 1) + '.csv')
            for t in range(0, 20000):  # Fetch data of per time-stamp
                real_time = t / 10
                df_now = man_file[(man_file['Tid'] < real_time) & (man_file['Tid'] >= real_time - 0.1)]
                if not df_now.empty:
                    count += 1
                    label_now = label_file[np.abs(label_file['Time'] - real_time) < 3 ] if \
                      not label_file[np.abs(label_file['Time'] - real_time) < 3 ].empty else label_file.iloc[[-1, -2], :]
                    X, Y, Z = df_now['X'].mean(), df_now['Y'].mean(), df_now['Z'].mean()
                    label = label_now['GroupID'].iloc[0]
                    info.append([i + 1, X, Y, Z, label])
    df = pd.DataFrame(info)
    df.columns = ['People', 'acceX', 'acceY', 'acceZ', 'label']
    print(df.describe())
    df.to_csv(label_folder + 'data.csv')

def classify():
    data_ = pd.read_csv(label_folder + 'data.csv')
    data_ = data_.sample(frac=0.1, random_state=1)  # sample
    data_.fillna(0, inplace=True)
    print(data_.describe())
    data = data_[['acceX', 'acceY', 'acceZ']]
    label = data_['label']
    data = preprocessing.MinMaxScaler().fit_transform(data)  
    model1 = KNeighborsClassifier()
    scores1 = cross_val_score(model1, data, label, cv=10, scoring='accuracy')
    model2 = RandomForestClassifier(n_estimators=200)
    scores2 = cross_val_score(model2, data, label, cv=10, scoring='accuracy')
    print(scores1, scores2, '\n', np.mean(scores1), np.mean(scores2))

# data_prepare()
classify()
