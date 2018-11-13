# coding:utf-8
'''
@time:    Created on  2018-11-03 12:51:18
@author:  Lanqing
@Func:    EEG_dataset.Explore_mat
'''

mat_path = 'E:/DATA/EEG_dataset/matFiles/'

#### Read mat files
import scipy.io
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

events = scipy.io.loadmat(mat_path + 'P04-events.mat')['data']
print(events, '\n', events.shape, '\n')

imported = []
for i in range(len(events)):
    a = ('type', events[i, 0], 'latency', events[i, 1], 'urevent', '')
    print(a)
    imported.append(a)
    

data = scipy.io.loadmat(mat_path + 'P04-eegdata.mat')['data']
print(data, '\n', data.shape)
 
a = data.T[:, -1]
plt.plot(a)
plt.show()
 
data = data.T[:10000, [1, 15, 29, 36, 12, 45, 33, 23]]
# plot_data = pd.DataFrame(data)
# print(plot_data.shape)
# plot_data.plot(subplots=True)
# plt.show()


