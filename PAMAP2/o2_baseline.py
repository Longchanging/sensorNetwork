# coding:utf-8
'''
@time:    Created on  2018-11-01 15:18:12
@author:  Lanqing
@Func:    PAMAP2.baseline
'''

from PAMAP2.o0_config import tmp_path_base
import pandas as pd, os, numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

# load
all_man_info = np.loadtxt(tmp_path_base + 'EarlyFusion.csv')
print('loaded data shape:', all_man_info.shape, \
      'label categories:', np.max(all_man_info[:, -1]))

# shuffle
idx = np.random.randint(all_man_info.shape[0], size=100000)
shuffled_data = all_man_info[idx]

# again fillna
shuffled_data = pd.DataFrame(shuffled_data).fillna(method='ffill').values

# fetch
data = shuffled_data[:, :-1]
label = shuffled_data[:, -1]

# maxmin
data = MinMaxScaler().fit_transform(data)

# baseline
model2 = RandomForestClassifier(n_estimators=100)

#### 10-fold cross validation
# scores2 = cross_val_score(model2, data, label, cv=10, scoring='accuracy', verbose=1)
# print(scores2)

# #### confusion matrix
data = MinMaxScaler().fit_transform(data)
# pca = PCA(n_components=2)
# data = pca.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1, shuffle=True) 
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
s1 = metrics.accuracy_score(y_test, y_pred)
f2 = metrics.confusion_matrix(y_test, y_pred)
f3 = metrics.f1_score(y_test, y_pred, average='macro')
print('\nAccuracy', s1, '\nConfusionMatrix\n', f2, '\nF1Score', f3) 
