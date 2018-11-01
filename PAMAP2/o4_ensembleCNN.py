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
from keras.layers import BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

#### CNN parameters
epochs = 4  # >>> should be 25+
lrate = 0.0001
decay = lrate / epochs
seed = 7
batch_size = 21  # channel as batch size

input_shape = (15, 20, 1)
nClasses = 12

def load_data():
    all_info = np.loadtxt(tmp_path_base + 'EarlyFusion.csv')
    return all_info

def goodModel():
    
    model = Sequential()
    model.add(Reshape(target_shape=(15, 20, 1), input_shape=(15, 20)))
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
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
    model.add(Reshape(target_shape=(15, 20, 1), input_shape=(15, 20)))
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
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

def sensorBasedEarlyFusion():
    return

# create our CNN model

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
model = goodModel()

# check NaN
l1 = np.argwhere(np.isnan(X_train))
l2 = np.argwhere(np.isnan(X_test))
print('There are Nans: ', l1.shape, l2.shape)

# fit and run our model
np.random.seed(0)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])   
model.fit(X_train, y_train, nb_epoch=100, batch_size=2000)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
