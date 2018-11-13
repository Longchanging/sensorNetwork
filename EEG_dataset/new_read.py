# coding:utf-8
'''
@time:    Created on  2018-11-08 10:08:07
@author:  Lanqing
@Func:    EEG_dataset.new_read
'''

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import LSTM
from keras.layers.core import Dropout, Dense, Activation
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn import metrics 
from keras.layers import  Reshape, Flatten, Conv2D, MaxPooling2D, Merge, Dropout
from algorithms import *
import keras
import numpy as np
import scipy.io as sio
import os

# Path
input_path = 'E:/DATA/SEED/seed_data/Preprocessed_EEG/'  # Use processed data

# Data Scale control
window_size = 15
load_max_rows = 2000
use_person = 10

#### CNN parameters
epochs = 5  
input_shape = (15, 20, 1)
filter_num = 16
batch_size = 2000

def change_shape(data, label):  # Load data from files
    l, labels = [], []
    print(label)
    for key in data.keys():
        if '__' not in key:
            name = int(key.split('eeg')[1])
            corrsponding_label = label[name - 1] + 1  # translate from [-1,0,1] to [0,1,2]
            d = data[key].T
            rows, channels = d.shape
            d = MinMaxScaler().fit_transform(d)
            clips = int(rows / window_size)
            use_rows = load_max_rows if load_max_rows < clips else clips
            d = np.reshape(d[:use_rows * window_size], (use_rows, window_size, channels))
            tmp_label = corrsponding_label * np.ones([use_rows , 1])
            print(d.shape)
            l.extend(d)
            labels.extend(tmp_label)
    one_data = np.array(l).reshape(-1, window_size, channels)
    one_label = np.array(labels).reshape(-1, 1)
    print('One person Shapes:', one_data.shape, one_label.shape)
    return one_data, one_label

def load_all_data():
    all_data, all_label = [], []
    i = 1
    label = sio.loadmat(input_path + 'label.mat')['label'][0]
    for _, _, c in os.walk(input_path):
        for item in c:
            if '_' in item and i < use_person:
                print('File %s included' % item)
                tmp_file = sio.loadmat(input_path + item)
                one_person_data, one_person_label = change_shape(tmp_file, label)
                all_data.append(one_person_data)
                all_label.append(one_person_label)
                i += 1
    all_data = vstack_list(all_data)
    
    
    # Min max scaler
    # reshaped = all_data.reshape([all_data.shape[0] * all_data.shape[1], all_data.shape[2]])
    # print('prepared for minmaxed and FFT from shape: ', reshaped.shape, 'to shape', reshaped.shape)
    # FFT
    # reshaped = fft_transform(reshaped)
    # minmaxed = MinMaxScaler().fit_transform(reshaped)
    # all_data = minmaxed.reshape([all_data.shape[0] , all_data.shape[1], all_data.shape[2]])    
    # print('after minmaxed shape:', all_data.shape)
    
    all_label = vstack_list(all_label)
    print('Final data shape', all_data.shape, all_label.shape)
    return all_data, all_label

def finalModel():
    model = Sequential()
    model.add(Reshape(target_shape=(62, window_size, 1), input_shape=(window_size, 62)))
    model.add(Conv2D(filter_num, (10, 5), activation='relu'))
    model.add(Conv2D(filter_num, (10, 5), activation='relu'))  
    model.add(Conv2D(filter_num, (10, 5), activation='relu'))  
    model.add(Conv2D(filter_num, (1, 1), activation='relu'))  
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(3, activation='softmax')) 
    model.summary()
    return model

data, label = load_all_data()
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1, shuffle=True)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.1, random_state=3, shuffle=True)
print('Data shapes: train,test,validate: \n', X_train.shape, X_test.shape, X_validate.shape, y_train.shape, y_test.shape, y_validate.shape)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
y_validate = keras.utils.to_categorical(y_validate)

model = finalModel()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
 
model.fit(X_train, y_train, epochs=epochs, validation_data=(X_validate, y_validate), verbose=1, batch_size=batch_size)
re = model.predict(X_test)

actual_y_list, prediction_y_list = [], []
for item in y_test:
    actual_y_list.append(np.argmax(item))
     
for item in re:
    prediction_y_list.append(np.argmax(item))
 
acc = get_acc(re, y_test)
cm = metrics.confusion_matrix(actual_y_list, prediction_y_list)

Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(prediction_y_list, actual_y_list) 
print(acc)
print(Precise, '\n', Recall, '\n', F1Score, '\n', Micro_average, accuracy_all)
print('\n', cm) 

model.save(str('all') + '.h5')
with open(str('all') + '.txt', 'w') as f:
    f.write('Accuracy:' + str(acc))
