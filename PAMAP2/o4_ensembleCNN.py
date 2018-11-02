# coding:utf-8
'''
@time:    Created on  2018-10-30 15:24:59
@author:  Lanqing
@Func:    CNN_Fusion.src.ensembleCNN
'''

import numpy as np
from PAMAP2.o0_config import tmp_path_base, window_size
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.layers import Merge
from keras.layers import BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import keras.layers.merge as merge
import time

#### CNN parameters
epochs = 10  
input_shape = (15, 20, 1)
filter_num = 16
nClasses = 12
batch_size = 3000

def load_data():
    all_info = np.loadtxt(tmp_path_base + 'EarlyFusion.csv')
    return all_info

def goodModel():
    
    model = Sequential()
    model.add(Reshape(target_shape=(15, window_size, 1), input_shape=(15, window_size)))
    
    model.add(Conv2D(filter_num, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(filter_num, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax')) 
    model.summary()

    return model

def EarlyFusion():
    
    model = Sequential()
    model.add(Reshape(target_shape=(15, window_size, 1), input_shape=(15, window_size)))
    
    model.add(Conv2D(filter_num, (15, 3), activation='relu', input_shape=input_shape))
    # model.add(Conv2D(filter_num, (1, 3), activation='relu'))
  
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nClasses, activation='softmax')) 
    model.summary()

    return model

def ShareFilter():
    
    model = Sequential()
    model.add(Reshape(target_shape=(15, window_size, 1), input_shape=(15, window_size)))
    
    model.add(Conv2D(filter_num, (1, 3), activation='relu', input_shape=input_shape))
    # model.add(Conv2D(filter_num, (1, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nClasses, activation='softmax')) 
    model.summary()

    return model

def channelBased(channel_num):
    
    channel_list = []
    
    for _ in range(channel_num): 
        one_channel = Sequential()
        one_channel.add(Reshape(target_shape=(1, window_size, 1), input_shape=(1, window_size)))
        one_channel.add(Conv2D(filter_num, (1, 3), activation='relu', input_shape=input_shape))
        one_channel.summary()
        channel_list.append(one_channel)
    
    merged = Merge(channel_list, mode='concat')
    
    model = Sequential()
    model.add(merged)    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nClasses, activation='softmax')) 
    model.summary()

    return model  

def ourMethod():
    return  

def sensorBasedEarlyFusion():    
    
    num_sensors = [2, 3, 3, 3, 4]
    
    branch1 = Sequential()
    branch1.add(Reshape(target_shape=(num_sensors[0], window_size, 1), input_shape=(num_sensors[0], window_size)))
    branch1.add(Conv2D(filter_num, (num_sensors[0], 3), activation='relu', input_shape=input_shape))
    branch1.summary()
    
    branch2 = Sequential()
    branch2.add(Reshape(target_shape=(num_sensors[1], window_size, 1), input_shape=(num_sensors[1], window_size)))
    branch2.add(Conv2D(filter_num, (num_sensors[1], 3), activation='relu', input_shape=input_shape))
    branch2.summary()

    branch3 = Sequential()
    branch3.add(Reshape(target_shape=(num_sensors[2], window_size, 1), input_shape=(num_sensors[2], window_size)))
    branch3.add(Conv2D(filter_num, (num_sensors[2], 3), activation='relu', input_shape=input_shape))
    branch3.summary()
        
    branch4 = Sequential()
    branch4.add(Reshape(target_shape=(num_sensors[3], window_size, 1), input_shape=(num_sensors[3], window_size)))
    branch4.add(Conv2D(filter_num, (num_sensors[3], 3), activation='relu', input_shape=input_shape))
    branch4.summary()
    
    branch5 = Sequential()
    branch5.add(Reshape(target_shape=(num_sensors[4], window_size, 1), input_shape=(num_sensors[4], window_size)))
    branch5.add(Conv2D(filter_num, (num_sensors[4], 3), activation='relu', input_shape=input_shape))
    branch5.summary()
        
    merged = Merge([branch1, branch2, branch3, branch4, branch5], mode='concat')
    
    model = Sequential()
    model.add(merged)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nClasses, activation='softmax')) 
    model.summary()
    
    return model


# read Early Fusion data
all_info = load_data()
data = all_info[:, :-1]
rows, cols = data.shape
data = data.reshape([rows, int(cols / window_size), window_size])  # reshape
label = all_info[:, -1]

# split data
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1, shuffle=True) 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print('Shapes: ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print('Info:', np.std(X_train))

# prepare sensor based data
sensor_heart_tempreture_train, sensor_heart_tempreture_test = X_train[:, [0, 1], :], X_test[:, [0, 1], :]
sensor_3d_acc_16g_train, sensor_3d_acc_16g_test = X_train[:, [2, 3, 4], :], X_test[:, [2, 3, 4], :]
sensor_3d_acc_6g_train, sensor_3d_acc_6g_test = X_train[:, [5, 6, 7], :], X_test[:, [5, 6, 7], :]
sensor_gyroscope_train, sensor_gyroscope_test = X_train[:, [8, 9, 10], :], X_test[:, [8, 9, 10], :]
other_train, other_test = X_train[:, 11:, :], X_test[:, 11:, :]

# prepare channel based data
train_list_channels, test_list_channels = [], []
for i in range(int(cols / window_size)):
    train_list_channels.append(X_train[:, i, :].reshape([-1, 1, window_size]))
    test_list_channels.append(X_test[:, i, :].reshape([-1, 1, window_size]))

# train and calculate time used
start_time = time.time()
np.random.seed(0)

# Early Fusion
print('\nEarly Fusion Model Started\n')
model = EarlyFusion()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
end_time1 = time.time()  # Show Info
print("EF Accuracy: %.2f%%" % (scores[1] * 100))
print("EF time used for %d epochs with batch size %d : %.3f" % (epochs, batch_size, end_time1 - start_time))
 
# Share Filter
print('\nShare Filter Model Started\n')
model = ShareFilter()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
end_time2 = time.time()  # Show Info
print("SF Accuracy: %.2f%%" % (scores[1] * 100))
print("SF time used for %d epochs with batch size %d : %.3f" % (epochs, batch_size, end_time2 - end_time1))
 
# Sensor based
print('\nSensor Based Model Started\n')
model = sensorBasedEarlyFusion()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
model.fit(x=[sensor_heart_tempreture_train, sensor_3d_acc_16g_train, sensor_3d_acc_6g_train, sensor_gyroscope_train, other_train], \
          y=y_train, nb_epoch=epochs, batch_size=batch_size, verbose=2)
scores = model.evaluate(x=[sensor_heart_tempreture_test, sensor_3d_acc_16g_test, sensor_3d_acc_6g_test, sensor_gyroscope_test, other_test], y=y_test, verbose=0)
end_time3 = time.time()  # Show Info
print("SB Accuracy: %.2f%%" % (scores[1] * 100))
print("SB Time used for %d epochs with batch size %d : %.3f" % (epochs, batch_size, end_time3 - end_time2))  # end_time2

# Channel based
print('\nChannel Based Model Started\n')
model = channelBased(int(cols / window_size))  # int(cols / window_size)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
model.fit(x=train_list_channels, y=y_train, nb_epoch=epochs, batch_size=batch_size, verbose=2)
scores = model.evaluate(x=test_list_channels, y=y_test, verbose=0)
end_time4 = time.time()  # Show Info
print("CB Accuracy: %.2f%%" % (scores[1] * 100))
print("CB Time used for %d epochs with batch size %d : %.3f" % (epochs, batch_size, end_time4 - end_time3))  # end_time3