# coding:utf-8
'''
@time:    Created on  2018-10-31 21:05:17
@author:  Lanqing
@Func:    PAMAP2.prepare
'''

from PAMAP2.o0_config import tmp_path_base, window_size, use_rows
from collections import Counter
import pandas as pd, os, numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def vstack_list(tmp):
    if len(tmp) > 1:
        data = np.vstack((tmp[0], tmp[1]))
        for i in range(2, len(tmp)):
            data = np.vstack((data, tmp[i]))
    else: data = tmp[0]    
    return data

# load
all_man_info = np.loadtxt(tmp_path_base + 'all_data.csv', delimiter=',', \
                          skiprows=0)  # 2800000 in total

# shuffle
idx = np.random.randint(all_man_info.shape[0], size=use_rows)
shuffled_data = all_man_info[idx]

# check
max_label, unique_labels = np.max(shuffled_data[:, -1]), np.unique(shuffled_data[:, -1])
print('\n loaded data shape:', all_man_info.shape, \
      '\n label categories:', unique_labels,
      '\n max label:', max_label,
      '\n count:', Counter(all_man_info[:, -1]))

# again fillna
shuffled_data = pd.DataFrame(shuffled_data).fillna(method='ffill').values

# fetch
datas = shuffled_data[:, :-1]
labels = shuffled_data[:, -1]
rows, cols = datas.shape

#################   Prepare different data for CNN  ###################

#### Early fusion

list_all_data, list_all_label = [], []
for label in unique_labels:
    if label != 0.0:
        loc = np.where(labels == label)[0]  # find label i data
        print('label:', label, 'length:', len(loc))
        index_list = loc if len(loc) % window_size == 0 \
            else loc[:(len(loc) - len(loc) % window_size)] 
        per_label_data = datas[index_list]
        per_label = labels[index_list]
        per_label_data = per_label_data.reshape([int(len(index_list) \
                                / window_size), window_size * cols])  # reshape as window size
        per_label = per_label.reshape([int(len(index_list) \
                                / window_size), window_size])[:, 0]  # reshape as window size
        per_label = per_label.reshape([len(per_label), 1])
        list_all_data.append(per_label_data)
        print(per_label.shape)
        list_all_label.append(per_label)
    
all_data = vstack_list(list_all_data)  # finish and process
all_data = MinMaxScaler().fit_transform(all_data)  # min max
all_label = vstack_list(list_all_label)   
le = LabelEncoder()
all_label = np.array(le.fit_transform(all_label))  # label encoder
all_label = all_label.reshape([len(all_label), 1])
all_info = np.hstack((all_data, all_label))
np.savetxt(tmp_path_base + 'EarlyFusion.csv', all_info)    
