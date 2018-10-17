# coding:utf-8
'''
@time:    Created on  2018-10-12 23:48:02
@author:  Lanqing
@Func:    src.read_group_labels
'''
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from src.config import  sigma, overlap_window, window_length, \
    model_folder, train_data_rate, train_folders, use_feature_type, train_folder
from src.functions import gauss_filter, fft_transform, divide_files_by_name, read_single_txt_file_new, \
    min_max_scaler, one_hot_coding, PCA, \
    train_test_evalation_split, knn_classifier, random_forest_classifier, validatePR, check_model, generate_configs, \
    vstack_list
    
def calcu_similarity(list1, list2):
    #### two list may have different length
    #### calculate the similarities
    l1 = len(list1)
    l2 = len(list2)
    count_equal1, count_equal2 = 0, 0
    for item1 in list1:
        for item2 in list2:
            if int(item1) == int(item2):
                count_equal1 += 1
    for item2 in list2:
        for item1 in list1:
            if int(item2) == int(item1):
                count_equal2 += 1
    return (count_equal1 * count_equal2) / (l1 * l2)

file_ = train_folder + '/annotation/fformationGT.csv'

final_groups = OrderedDict()
g1 = [9, 11, 16, 10, 14, 18]
g2 = [15, 1 ]               
g3 = [2, 13, 7, 3, 6, 5]
g4 = [8, 12, 17, 4]        
groups = {1: g1,
          2:g2,
          3:g3,
          4:g4
          }
tmp_group = groups

fid = open(file_, 'r')
previous_time = 0.1
count_line = 0

for line in fid:
    if line:
        
        count_line += 1
        line = line.strip('\n').split(',')
        time_now = line[0]
        data = line[1:]
        
        #### remove extra ','
        ttmp = []
        for item in data:
            if item:
                ttmp.append(int(item))
        data = ttmp
        
        #### fetch data every 4 line 
        #### And group data into the four group according the 18 person
        max_score, most_similar_group = 0, []
        for key in groups.keys():
            new_score = calcu_similarity(data, groups[key])
            if  new_score > max_score:
                max_score = new_score
                max_index = key  #### get group id 
                most_similar_group = groups[key]
                
        tmp_group[max_index] = data
        
        # print(data, groups[key])
        
        if  previous_time != time_now:
            print(time_now)
            final_groups[time_now] = tmp_group
            previous_time = time_now
            print(final_groups[time_now])
            
print(final_groups)

#### Find the appear group of per person of all the time
dict_all_person = {}
dict_per_person = {}
list_tmp = []
for i in range(19):  # initial
    dict_all_person[i + 1] = {}
  
person_dict, person_list = {}, []
for i in range(18):  # person
    per_person_dict = OrderedDict()
    for key in final_groups.keys():  # time
        for per_group in final_groups[key].keys():  # group list
            # print(final_groups[key][per_group])
            if (i + 1) in final_groups[key][per_group]:
                # print(final_groups[key][per_group])
                per_person_dict[key] = per_group
                person_list.append([key, per_group])
    df = pd.DataFrame.from_dict(per_person_dict, orient='index')
    df.to_csv('../data/tmp/' + str(i + 1) + '.csv')
    # person_dict[(i + 1)] = person_list

# #### Find the appear group of per person of all the time
# dict_all_person = {}
# dict_per_person = {}
# list_tmp = []
# for i in range(19):  # initial
#     dict_all_person[i + 1] = {}
#     
# for key in final_groups.keys():
#     print('Hello,', key, final_groups[key])
#     for per_group in final_groups[key].keys():
#         a = []
#         for i in range(19):
#             if i + 1 in final_groups[key][per_group]:
#                 a.append([key, per_group])
#                 dict_all_person[i + 1] = a
# print(time_now, max_score, max_index)
