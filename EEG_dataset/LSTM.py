#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author： Liang Fang
# Writen for NNet hw4 problem 1: LSTM network for SEED dataset

import keras
from keras.layers import LSTM
from keras.layers.core import Dropout, Dense, Activation
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.preprocessing import MinMaxScaler
from algorithms import *
import numpy as np

import numpy as np
import scipy.io as sio
import os

# Path
input_path = 'E:/DATA/SEED/seed_data/Preprocessed_EEG/'  # Use processed data

# LSTM parameters
TimeStep = 1  # 10
HiddenLayers = 64
EPOCHS = 1000
BatchSize = 24
validate_ratio = 0.1

#### CNN parameters
epochs = 10  
input_shape = (15, 20, 1)
filter_num = 16
nClasses = 12
batch_size = 3000
choose_channel_number = 15  # total used channel number
loop_number = 100

# Data Scale control
window_size = 10
load_max_rows = 500
use_person = 2

# Load data from files
def npzload(data):
    l = []
    for key in data.keys() :
        if  '__' not in key:
            d = data[key]
            # d = np.transpose(d, (1, 0, 2))
            d = np.transpose(d, (1, 0))
            print(type(d), d.shape)
            # print(d.shape)
            # d = np.reshape(d, (d.shape[0], d.shape[1] * d.shape[2]))
            l.append(d)
            # print(d.shape)
    return l

def load_all_data():
    data = {}
    i = 1
    for _, _, c in os.walk(input_path):
        for item in c:
            if '_' in item and i < use_person:
                print('File %s included' % item)
                tmp_file = sio.loadmat(input_path + item)
                tmp_file = npzload(tmp_file)
                data[i] = tmp_file
                i += 1
    label = sio.loadmat(input_path + 'label.mat')['label']
    data['label'] = keras.utils.to_categorical(label)
    return data

def compress(data, label):
    x = []
    y = []
    for i in range(len(data)):
        clip = data[i]
        tmp = int(clip.shape[0] / TimeStep)
        for step in range(tmp):
            l = step * TimeStep
            r = l + TimeStep
            x.append(np.reshape(clip[l:r], (TimeStep, clip.shape[1])))
            y.append(label[i])
    return np.array(x), np.array(y)


def preprocess(data, label):
    tmpall = []
    for clip in data:
        tmpall.extend(clip)
        print(clip.shape)
    # tmpall = np.array(tmpall).reshape(clip.shape[0], -1)
    tmpall = np.array(tmpall)
    print(type(tmpall), tmpall.shape)
    scaler = MinMaxScaler()
    scaler.fit(tmpall)
    re = []
    for clip in data:
        re.append(scaler.transform(clip))
    # train, test = (compress(re[:9], label[:9]), compress(re[9:], label[9:]))
    train, test = (compress(re[:9], label[:9]), compress(re[9:], label[9:]))
    
    np.savetxt('FL_train.txt', train[0].reshape([ train[0].shape[0] * train[0].shape[1], train[0].shape[2]]))
    
    # Shuffle
    index = list(range(train[0].shape[0]))
    np.random.shuffle(index)
    shuffle_train = (train[0][index], train[1][index])
    index = list(range(test[0].shape[0]))
    np.random.shuffle(index)
    shuffle_test = (test[0][index], test[1][index])
    return (shuffle_train, shuffle_test)

def get_data(index):
    data = load_all_data()
    curdata = preprocess(data[index], data['label'])
    return (compress(curdata[:9], data['label'][:9]), compress(curdata[9:], data['label'][9:]))

def concentrate(data):
    return

def get_model():
    model = Sequential()
    model.add(LSTM(input_shape=(TimeStep, 310), units=16))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def train_all():
    
    train_datas, train_labels, test_datas, test_labels = [], [], [], []
    
    for i in range(1, 4):
        train, test = get_data(i)
        train_data, train_label, test_data, test_label = train[0], train[1], test[0], test[1]
        print(train_data.shape)
        train_datas.append(train_data) 
        train_labels.append(train_label) 
        test_datas.append(test_data) 
        test_labels.append(test_label) 

    train_data = vstack_list(train_datas)  # concentrate data
    train_label = vstack_list(train_labels)
    test_data = vstack_list(test_datas)
    test_label = vstack_list(test_labels)
    
    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    model = get_model()
    
    X_train, X_validate, y_train, y_validate = train_test_evalation_split(train_data, train_label, validate_ratio=validate_ratio)
    print(X_train.shape, y_train.shape)

    model.fit(X_train, y_train, epochs=300, validation_data=(X_validate, y_validate), verbose=1)
    re = model.predict(test_data)
    
    actual_y_list, prediction_y_list = [], []
    for item in test_label:
        actual_y_list.append(np.argmax(item))
    print(actual_y_list)
        
    for item in re:
        prediction_y_list.append(np.argmax(item))
    print(prediction_y_list)
    
    acc = get_acc(re, test_label)
    Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(prediction_y_list, actual_y_list) 
    print(acc)
    print(Precise, '\n', Recall, '\n', F1Score, '\n', Micro_average, accuracy_all)
    
    model.save(str('all') + '.h5')
    with open(str('all') + '.txt', 'w') as f:
        f.write('Accuracy:' + str(acc))

    return

def train(index):
    model = get_model()
    train, test = get_data(index)
    X_train, X_validate, y_train, y_validate = train_test_evalation_split(train[0], train[1], validate_ratio=validate_ratio)
    print(train[0].shape, test[0].shape)

    model.fit(X_train, y_train, epochs=300, validation_data=(X_validate, y_validate), verbose=1)
    re = model.predict(test[0])
    
    import numpy as np
    actual_y_list, prediction_y_list = [], []
    for item in test[1]:
        actual_y_list.append(np.argmax(item))
    print(actual_y_list)
        
    for item in re:
        prediction_y_list.append(np.argmax(item))
    print(prediction_y_list)
    
    acc = get_acc(re, test[1])
    Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(prediction_y_list, actual_y_list) 
    print(acc)
    print(Precise, '\n', Recall, '\n', F1Score, '\n', Micro_average, accuracy_all)
    
    model.save(str(index) + '.h5')
    with open(str(index) + '.txt', 'w') as f:
        f.write('Accuracy:' + str(acc))
        
    return
        
if __name__ == '__main__':
    # for i in range(1, 4):
    #    train(i)
        
    # problem2
    train_all()

