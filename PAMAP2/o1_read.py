# coding:utf-8
'''
@time:    Created on  2018-10-31 19:18:44
@author:  Lanqing
@Func:    PAMAP2.new_read
'''
from PAMAP2.o0_config import input_path_base, tmp_path_base, \
    pid, data_columns, label_columns
import pandas as pd, os, numpy as np

def vstack_list(tmp):
    if len(tmp) > 1:
        data = np.vstack((tmp[0], tmp[1]))
        for i in range(2, len(tmp)):
            data = np.vstack((data, tmp[i]))
    else: data = tmp[0]    
    return data

all_man_data_list, all_man_label_list = [], []
for item in pid:  # pid  # one person
    file2read = input_path_base + 'subject' + str(item) + '.dat'
    if os.path.exists(file2read):
        single_man_df = pd.read_csv(file2read, delimiter=' ')
        single_man_data = single_man_df.iloc[:, data_columns]
        single_man_data = single_man_data.fillna(method='ffill')
        single_man_label = single_man_df.iloc[:, label_columns]
        all_man_data_list.append(single_man_data)
        all_man_label_list.append(single_man_label)
        print(single_man_data.describe())

all_man_data = vstack_list(all_man_data_list)
all_man_label = vstack_list(all_man_label_list)
all_man_info = np.hstack((all_man_data, all_man_label))
np.savetxt(tmp_path_base + 'all_data.csv', all_man_info, delimiter=',')
