# coding:utf-8
'''
@time:    Created on  2018-10-19 06:02:12
@author:  Lanqing
@Func:    src.wav
'''
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import pandas as pd, numpy as np

fs1, data1 = wavfile.read('C:/Users/jhh/Desktop/audio/REC20181019054833.wav')
fs2, data2 = wavfile.read('C:/Users/jhh/Desktop/audio/REC20181019055029.wav')

df1 = pd.DataFrame(data1).iloc[:, 0]
df2 = pd.DataFrame(data2).iloc[:, 0]

df1.plot()
plt.show()
df2.plot()
plt.show()

n = 800
timestep = 1 / 48000

list_df1 = [list(df1[i:i + n].values) for i in range(0, df1.shape[0], n)][:-1]
list_df2 = [list(df2[i:i + n].values) for i in range(0, df2.shape[0], n)][:-1]

np1 = np.array(list_df1)
np2 = np.array(list_df2)

transformed = np.fft.fft(np1)  # FFT
np11 = np.abs(transformed)
# freq1 = np.fft.fftfreq(n, d=timestep)

transformed = np.fft.fft(np2)  # FFT
np21 = np.abs(transformed)

mean1, max1, min1, std1 = np1.mean(axis=1), np1.max(axis=1), np1.min(axis=1), np1.std(axis=1)
mean2, max2, min2, std2 = np2.mean(axis=1), np2.max(axis=1), np2.min(axis=1), np2.std(axis=1)
mean1, max1, min1, std1 = mean1.reshape([len(np1), 1]), max1.reshape([len(np1), 1]), min1.reshape([len(np1), 1]), std1.reshape([len(np1), 1])
mean2, max2, min2, std2 = mean2.reshape([len(np2), 1]), max2.reshape([len(np2), 1]), min2.reshape([len(np2), 1]), std2.reshape([len(np2), 1])

tmp1 = np.zeros([len(np1), 1]) 
tmp2 = np.ones([len(np2), 1]) 
# freq2 = np.fft.fftfreq(n, d=timestep)

# d1 = np.hstack((np1, np11, tmp1))
# d2 = np.hstack((np2, np21, tmp2))
print(np11.shape, mean1.shape, max1.shape, min1.shape, std1.shape, tmp1.shape)
d1 = np.hstack((np11, mean1, max1, min1, std1, tmp1))
d2 = np.hstack((np21, mean2, max2, min2, std2, tmp2))

data = np.vstack((d1, d2))
label = data[:, -1]
data = data[:, :-1]

data = preprocessing.MinMaxScaler().fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1, shuffle=True) 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) 
model2 = RandomForestClassifier(n_estimators=200)  # (max_depth=5, min_samples_leaf=5000)  # (max_depth=5, max_leaf_nodes=10)
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)

s1 = metrics.accuracy_score(y_test, y_pred)
f2 = metrics.confusion_matrix(y_test, y_pred)
print('\nAccuracy', s1, '\nConfusionMatrix\n', f2)