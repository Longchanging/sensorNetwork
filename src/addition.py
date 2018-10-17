# coding:utf-8
'''
@time:    Created on  2018-10-17 16:55:24
@author:  Lanqing
@Func:    Developed upon the simple version
'''

import os
import pandas as pd, numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from src.config import  train_data_rate, train_folders
from src.functions import generate_configs

input_folder, label_folder = '../data/input/scene_present/accel_fast/', '../data/tmp/'

def divide_files_by_name(folder_name, different_category):
    #### sensor folder -> file dict
    dict_file = dict(zip(different_category, [[]] * len(different_category)))  # initial
    for category in different_category:
        dict_file[category] = []  # Essential here
        for (root, tmp, files) in os.walk(folder_name):  # List all file names
            for filename in files:  # Attention here
                if not tmp and category in root:  # not a single file, and is a folder
                    per_file = os.path.join(root, filename)
                    dict_file[category].append(per_file)
    return dict_file

def read_single_txt_file_new(single_file_name):
    df = pd.read_csv(single_file_name, sep=',', header=None)
    #### rename
    names = df.columns  
    file_name = single_file_name.split('/')[-1].split('\\')[0]
    new_name = []
    for item in names:
        if item == 0: 
            new_name.append('Time')
        else:
            new_name.append(str(file_name + '_' + str(item)))
    df.columns = new_name
    #### Filter empty info
    for column in df.columns:
        if np.all(df[column] == df[column][0]):
            df.drop(column, axis=1)
    # print(df.describe())
    return df

def read_all_df(input_folder, different_category, percent2read_afterPCA_data, after_fft_data_folder):
    '''
        Para : Read files according to the folder and keyword info.
        Return : Every files every keyword return corresponding files dataframe.
    '''
    file_dict = divide_files_by_name(input_folder, different_category)
    all_data_list = {}
    for category in different_category:
        files_list = {}
        for one_category_single_file in file_dict[category]:  # No worry "Ordered",for it is list
            file_array = read_single_txt_file_new(one_category_single_file)
            files_list[one_category_single_file] = file_array
        all_data_list[category] = files_list
    # print(all_data_list.keys(), files_list.keys())
    return all_data_list

def get_useful_df(all_df_list):
    all_person_info = []
    info_complete, complete_info_manID = {}, []
    #### fetch complete info person ID
    for i in  range(18):
        info_complete[i + 1] = 0
        tmp_dict = []
        for sensor_name in  all_df_list.keys():
            for sensor_per_man in all_df_list[sensor_name].keys():
                sensor_per_man_name = int(sensor_per_man.split('\\')[-1].split('.')[0])
                if (i + 1) == sensor_per_man_name:
                    info_complete[i + 1] += 1
                    tmp_dict.append(all_df_list[sensor_name][sensor_per_man])
                    if info_complete[i + 1] == 3:
                        complete_info_manID.append(i + 1)
                        all_person_info.append(tmp_dict)
                    # print(all_df_list[sensor_name][sensor_per_man])
    print('All person sensors num: ', info_complete, '\n Person with full info: ', complete_info_manID)  
    #### All person info only contains people with full info
    return all_person_info, complete_info_manID

def merge_one_person_data(all_person_info, PId_list):
    #### given one person several dataframe and the label folder
    #### return full dataframe one person 
    
    def count_no_empty(df_info_list):
        count_not_empty = 0
        for i in range(len(df_info_list)):
            if  not df_info_list[i].empty:
                count_not_empty += 1
        return count_not_empty
    
    info = []  #### target data format:  (people), columns,label
    personlistIndex = 0 
    count_all = 0
    
    print(PId_list, len(all_person_info))
    
    for pid in PId_list:
        one_person_data = all_person_info[personlistIndex]  # Get the corresponding person data
        total_sensors = len(one_person_data)
        personlistIndex += 1
        print('person: ', pid, 'countline: ', count_all)
        for t in range(0, 20000):  # Fetch data of per time-stamp
            
            count_sensors = 0
            df_list, df_info_list = [], []
            
            for one_sensor_dataframe in one_person_data:
                
                label_file = pd.read_csv(label_folder + str(pid) + '.csv')
                real_time = t / 10
                df_now = one_sensor_dataframe[(one_sensor_dataframe['Time'] < real_time) & (one_sensor_dataframe['Time'] >= real_time - 0.1)]
                df_info_list.append(df_now)
                count_sensors += 1
                                
                if count_sensors == total_sensors:    
                    count_not_empty = count_no_empty(df_info_list)         
                    if count_not_empty == len(df_info_list):
                        count_all += 1       
                        for i in range(len(df_info_list)):
                            #### get the dataframe
                            df_now = df_info_list[i]
                            df_now = df_now.drop(columns=['Time'])  # drop the time column
                            df_now = np.mean(df_now, axis=0)
                            #### get the mean values 
                            df_list.extend(list(df_now.values))
                            #### deal with labels
                        label_now = label_file[np.abs(label_file['Time'] - real_time) < 3 ] if \
                             not label_file[np.abs(label_file['Time'] - real_time) < 3 ].empty else label_file.iloc[[-1, -2], :]
                        label = label_now['GroupID'].iloc[0] 
                        df_list.append(label)
                        info.append(df_list) 
                        # print(df_list[-5:])      
                           
    df = pd.DataFrame(info)
    df.to_csv(label_folder + 'All_data.csv')    
    return df

def prepare_data():
    #### 生成参数
    train_keyword, train_folder, _, _, train_tmp, _, _, _, model_folder, _ = generate_configs(train_folders, 'scene_present')
    df_list = read_all_df(train_folder, train_keyword, train_data_rate, train_tmp)  #### 读数据
    all_person_info, PId_list = get_useful_df(df_list)
    merge_one_person_data(all_person_info, PId_list)
    return

def classify():
    ##### classify using KNN and RF
    data_ = pd.read_csv(label_folder + 'All_data.csv')
    print(data_.describe())
    data_ = data_.sample(frac=0.5, random_state=1)  # sample
    data_.fillna(0, inplace=True)
    print(data_.describe())
    data = data_.iloc[:, :-1]
    label = data_.iloc[:, -1]
    data = preprocessing.MinMaxScaler().fit_transform(data)  
    model1 = KNeighborsClassifier()
    scores1 = cross_val_score(model1, data, label, cv=10, scoring='accuracy')
    model2 = RandomForestClassifier(n_estimators=200)
    scores2 = cross_val_score(model2, data, label, cv=10, scoring='accuracy')
    print(scores1, scores2, '\n', np.mean(scores1), np.mean(scores2))
    
prepare_data()
classify()
