# coding:utf-8
'''
@time:    Created on  2018-10-15 11:38:47
@author:  Lanqing
@Func:    src.rewrite_labels
'''

import pandas as pd
from collections import OrderedDict
from config import train_folder    

file_ = train_folder + '/annotation/fformationGT.csv'
final_groups, tmp_group = OrderedDict(), {}
fid = open(file_, 'r')
previous_time = 0

for line in fid:
    
    if line:
        
        #### fetch line
        line = line.strip('\n').split(',')
        time_now = line[0]
        data = line[1:]
        
        #### remove extra ','
        ttmp = []
        for item in data:
            if item:
                ttmp.append(int(item))
        data = ttmp
        
        #### get group id
        if 1 in data:
            group_id = 1
        elif 14 in data:
            group_id = 2
        elif 17 in data:
            group_id = 3
        elif 2 in data:
            group_id = 4 
        else:
            group_id = 0
            
        #### fetch data
        tmp_group[group_id] = data        
        if  previous_time != time_now:
            # print(time_now)
            final_groups[time_now] = tmp_group
            previous_time = time_now
            print(time_now)
            # print(tmp_group, final_groups[time_now])
            print(final_groups[time_now])
            
# print(final_groups)
for key in final_groups.keys():  # time
    for per_group in final_groups[key].keys():  # group list
        # print(final_groups[key][per_group])
        pass
        # print(i + 1, final_groups[key][per_group], (i + 1) in final_groups[key][per_group])


#### Find the appear group of per person of all the time
# person_dict, person_list = {}, []
# for i in range(18):  # person
#     per_person_dict = OrderedDict()
#     for key in final_groups.keys():  # time
#         for per_group in final_groups[key].keys():  # group list
#             # print(key, per_group, final_groups[key], final_groups[key][per_group])
#             # print(i + 1, final_groups[key][per_group], (i + 1) in final_groups[key][per_group])
#             if (i + 1) in final_groups[key][per_group]:
#                 # print(per_group)
#                 per_person_dict[key] = per_group
#                 person_list.append([key, per_group])
#                 # print([key, per_group])
#     df = pd.DataFrame.from_dict(per_person_dict, orient='index')
#     df.to_csv('../data/tmp/' + str(i + 1) + '.csv')
