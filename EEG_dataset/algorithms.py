# coding:utf-8
'''
@time:    Created on  2018-05-02 18:29:19
@author:  Lanqing
@Func:    
'''
import numpy as np

def train_test_evalation_split(data, label, validate_ratio):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label, \
                                                        test_size=validate_ratio, random_state=0)
    return X_train, X_test, y_train, y_test

def fft_transform(vector):
    '''
        FFT transform if necessary, only save the real part
        @overwrite: Must Return Ampetitude
    '''
    transformed = np.fft.fft(vector)  # FFT
    transformed = transformed.reshape([transformed.shape[0] * transformed.shape[1], 1])  # reshape 
    return np.abs(transformed)

def vstack_list(tmp):
    data = np.vstack((tmp[0], tmp[1]))
    for i in range(2, len(tmp)):
        data = np.vstack((data, tmp[i]))
    return data

def hstack_list(tmp):
    data = np.hstack((tmp[0], tmp[1]))
    for i in range(2, len(tmp)):
        data = np.hstack((data, tmp[i]))
    return data

def max_min(X):
    from sklearn import preprocessing
    max_min_model = preprocessing.MinMaxScaler().fit(X)
    X = max_min_model.transform(X) 
    return max_min_model, X

def shuffle(data, labels):
    '''
    Return a shuffle of random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def validatePR(prediction_y_list, actual_y_list):

    ''' 
    
        function : make 3 dictionaries to store and analysis predict result
        
        usage:  prediction_y_list -> label list predicted
                actual_y_list -> label list from original data
                
        return: 
                P ->   precision  , the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
                       true positives and ``fp`` the number of false positives. 
                R ->   recall score,the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
                       true positives and ``fn`` the number of false negative. 
                F1Score -> F-measure means (A + 1) * precision * recall / (A ^ 2 * precision + recall),when A = 1
                       F1Score becomes 2*P*R / P+R
                Micro_average -> average of all ctgry F1Scores
                accuracy_all -> accuracy means 'TP+TN / TP+TN+FP+FN'
                
    '''
    right_num_dict = {}
    prediction_num_dict = {}
    actual_num_dict = {}

    Precise = {}
    Recall = {}
    F1Score = {}
    
    if len(prediction_y_list) != len(actual_y_list):
        raise(ValueError)    
    
    for (p_y, a_y) in zip(prediction_y_list, actual_y_list):
        
        if p_y not in prediction_num_dict:
            prediction_num_dict[p_y] = 0
        prediction_num_dict[p_y] += 1

        if a_y not in actual_num_dict:  # here mainly for plot 
            actual_num_dict[a_y] = 0
        actual_num_dict[a_y] += 1

        if p_y == a_y:  # basis operation,to calculate P,R,F1
            if p_y not in right_num_dict:
                right_num_dict[p_y] = 0
            right_num_dict[p_y] += 1
    
    for i in  np.sort(list(actual_num_dict.keys()))  : 
                
        count_Pi = 0  # range from a to b,not 'set(list)',because we hope i is sorted 
        count_Py = 0
        count_Ri = 0
        count_Ry = 0

        for (p_y, a_y) in zip(prediction_y_list, actual_y_list):
            
            
            if p_y == i:
                count_Pi += 1
                
                if p_y == a_y:                              
                    count_Py += 1
                    
            if a_y == i :
                count_Ri += 1
                
                if a_y == p_y:
                    count_Ry += 1    
        
        Precise[i] = count_Py / count_Pi if count_Pi else 0               
        Recall[i] = count_Ry / count_Ri if count_Ri else 0
        F1Score[i] = 2 * Precise[i] * Recall[i] / (Precise[i] + Recall[i]) if Precise[i] + Recall[i] else 0
    
    Micro_average = np.mean(list(F1Score.values()))
    
    lenL = len(prediction_y_list)
    sumL = np.sum(list(right_num_dict.values()))
    accuracy_all = sumL / lenL
        
    return Precise, Recall, F1Score, Micro_average, accuracy_all

def get_acc(re, label):
    sum = 0.0
    for i in range(len(re)):
        if np.argmax(re[i]) == np.argmax(label[i]):
            sum += 1.0
    return sum / len(re)
