# coding:utf-8
'''
@time:    Created on  2018-10-17 11:14:47
@author:  Lanqing
@Func:    src.simple_version
'''
import pandas as pd, numpy as np, os
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from src.functions import generate_configs, validatePR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import graphviz 

input_folder, label_folder = '../data/input/scene_present/accel_fast/', '../data/tmp/'

def data_prepare():
    #### target data format:  (people), X, Y, Z,label
    info = []
    count = 0
    for i in  range(18): 
        print('people : ', i, 'Lines of data: ', count)
        if os.path.exists(input_folder + str('%.2d' % (i + 1)) + '.csv'):
            man_file = pd.read_csv(input_folder + str('%.2d' % (i + 1)) + '.csv')  # read per person acce data
            man_file.columns = ['Tid', 'X', 'Y', 'Z']
            label_file = pd.read_csv(label_folder + str(i + 1) + '.csv')
            for t in range(0, 20000):  # Fetch data of per time-stamp
                real_time = t / 10
                df_now = man_file[(man_file['Tid'] < real_time) & (man_file['Tid'] >= real_time - 0.1)]
                if not df_now.empty:
                    count += 1
                    label_now = label_file[np.abs(label_file['Time'] - real_time) <= 1.5 ] if \
                      not label_file[np.abs(label_file['Time'] - real_time) <= 1.5 ].empty else label_file.iloc[[-1, -2], :]
                    X, Y, Z = df_now['X'].mean(), df_now['Y'].mean(), df_now['Z'].mean()  # calculate mean
                    label = label_now['GroupID'].iloc[0] 
                    info.append([i + 1, X, Y, Z, label])
    df = pd.DataFrame(info)
    df.columns = ['People', 'acceX', 'acceY', 'acceZ', 'label']
    print(df.describe())
    df.to_csv(label_folder + 'data.csv')

def classify_cross_validation():
    ##### classify using KNN and RF
    data_ = pd.read_csv(label_folder + 'data.csv')
    data_ = data_.sample(frac=0.1, random_state=1)  # sample
    data_.fillna(0, inplace=True)
    print(data_.describe())
    data = data_[['acceX', 'acceY', 'acceZ']]
    label = data_['label']
    data = preprocessing.MinMaxScaler().fit_transform(data)  
    model1 = KNeighborsClassifier()
    scores1 = cross_val_score(model1, data, label, cv=10, scoring='accuracy')
    model2 = RandomForestClassifier(n_estimators=200)
    scores2 = cross_val_score(model2, data, label, cv=10, scoring='accuracy')
    model3 = tree.DecisionTreeClassifier()
    scores3 = cross_val_score(model3, data, label, cv=10, scoring='accuracy')
    print('\nKNN,RF,DT\n', scores1, scores2, scores3, '\nMean Accuracy:', np.mean(scores1), np.mean(scores2), np.mean(scores2))
    
def classify__confusion_matrix():
    data_ = pd.read_csv(label_folder + 'data.csv')
    data_ = data_.sample(frac=0.5, random_state=1)  # sample
    data_.fillna(0, inplace=True)
    print(data_.describe())
        
    data = data_[['acceX', 'acceY', 'acceZ']]
    label = data_['label']
    data = preprocessing.MinMaxScaler().fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1, shuffle=True) 
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) 
    # model2 = KNeighborsClassifier()
    # model2 = RandomForestClassifier(n_estimators=200)
    model2 = tree.DecisionTreeClassifier()  # (max_depth=5, min_samples_leaf=5000)  # (max_depth=5, max_leaf_nodes=10)
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    Precise, Recall, F1Score, Micro_average, accuracy_all = validatePR(y_pred, y_test)
    print('\n........  Outputing the result ....\n')
    print('Precise, Recall, F1Score, Micro_average, accuracy_all:', '\nPrecise',
          Precise, '\nRecall', Recall, '\nF1Score' , F1Score, '\nMicro_average', Micro_average, '\nAccuracy',
          accuracy_all)
    s1 = metrics.accuracy_score(y_test, y_pred)
    f2 = metrics.confusion_matrix(y_test, y_pred)
    print('\nAccuracy', s1, '\nConfusionMatrix\n', f2)
    dot_data = tree.export_graphviz(model2, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("teacher") 
    from src.classify_boundary_plot import plot_decision_boundary
    plot_decision_boundary(model2, X=data, Y=label)
    plt.show()
    
def plot_3d_acce(data, label):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    # ax3D = fig.add_subplot(111, projection='3d')
    
    marker = ['.', 'x', '+', 'o', '*', '.', '1', '2', '_', 'D', 'd', '8', '3', '|', '4', 'p', 'X', 'P', 'o', '8', '*', '1', '.', 'x', '+', 'o', '2', '*', '.']
    color = ['blue', 'firebrick', 'cadetblue', 'black', 'pink']
    ctgy = int(max(label)) + 1
    for i in range(ctgy):
        tmp = np.where(label == i)[0]
        p = plt.scatter(data[tmp, 0], data[tmp, 1], color=color[i], label=str(i), s=1)  
        # p = ax3D.scatter(data[tmp, 0], data[tmp, 1], data[tmp, 2], marker=marker[i], color=color[i], label=str(i), s=3)  
    plt.legend(loc='best')  
    plt.xlabel('AcceX')
    plt.ylabel('AcceY')
    plt.show()
    return
    
def do_experiment():
    from sklearn.decomposition import PCA
    from src.classify_boundary_plot import plot_decision_boundary
    from matplotlib import pyplot as plt 
    pca = PCA(n_components=2)

    data_s = pd.read_csv(label_folder + 'data.csv')
    data_s = data_s.sample(frac=0.5, random_state=1)  # sample
    data_s.fillna(0, inplace=True)
    score = []
    for i in [7 , 11, 12]:  # range(18):
        data_ = data_s[data_s['People'] == (i + 1)]  
        if not data_.empty:      
            data = data_[['acceX', 'acceY', 'acceZ']]
            label = data_['label']
            data = preprocessing.MinMaxScaler().fit_transform(data)
            data = pca.fit_transform(data)
            
            X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1, shuffle=True) 
            print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) 
            # model2 = KNeighborsClassifier()
            model2 = KNeighborsClassifier()  # RandomForestClassifier(n_estimators=200)
            # model2 = tree.DecisionTreeClassifier()  # (max_depth=5, min_samples_leaf=5000)  # (max_depth=5, max_leaf_nodes=10)
            model2.fit(X_train, y_train)
            y_pred = model2.predict(X_test)
            s1 = metrics.accuracy_score(y_test, y_pred)
            score.append(s1)
            print(s1)
            plot_3d_acce(data, label)
            # print(data, label.values)
            plot_decision_boundary(model2, X=data, Y=label.values)
            plt.show()
    print(score, '\n', np.mean(score))
# data_prepare()
# classify_cross_validation()
classify__confusion_matrix()
# do_experiment()
