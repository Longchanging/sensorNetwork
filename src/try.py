# coding:utf-8
'''
@time:    Created on  2018-10-17 11:14:47
@author:  Lanqing
@Func:    src.simple_version
'''
import pandas as pd, sklearn.preprocessing, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
input_folder, label_folder = '../data/input/scene_present/accel_fast/', '../data/tmp/'
data_ = pd.read_csv(label_folder + 'data.csv')
data_ = data_.sample(frac=0.1, random_state=1)  # sample
data_.fillna(0, inplace=True)
print(data_.describe())
data = data_[['acceX', 'acceY', 'acceZ']]
label = data_['label']
data = data.values
# data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')

marker = ['.', 'x', '+', 'o', '*', '.', '1', '2', '_', 'D', 'd', '8', '3', '|', '4', 'p', 'X', 'P', 'o', '8', '*', '1', '.', 'x', '+', 'o', '2', '*', '.']
color = ['blue', 'firebrick', 'cadetblue', 'black', 'pink']
ctgy = int(max(label)) + 1
for i in range(ctgy):
    tmp = np.where(label == i)[0]
    # p = plt.scatter(data[tmp, 0], data[tmp, 1], color=color[i], label=str(i), s=1)  
    p = ax3D.scatter(data[tmp, 0], data[tmp, 1], data[tmp, 2], marker=marker[i], color=color[i], label=str(i), s=3)  
plt.legend(loc='best')  
plt.xlabel('AcceX')
plt.ylabel('AcceY')

plt.show()

