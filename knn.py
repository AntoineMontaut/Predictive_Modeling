'''Unit4 Lesson4: K-Nearest Neighbors'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

df = pd.read_csv('iris/iris.data.csv')
df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'iris']
features = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
# print(df.info())
# print(df['iris'].unique())
# print(df.iris.value_counts())

def plot_scatter():
    fig = plt.figure('Iris')
    (alpha, size) = (.7, 50)
    plt.scatter(df[df.iris=='Iris-setosa'].sepal_len, df[df.iris=='Iris-setosa'].sepal_wid, s=size, c='r', alpha=alpha, label='Setosa')
    plt.scatter(df[df.iris=='Iris-versicolor'].sepal_len, df[df.iris=='Iris-versicolor'].sepal_wid, s=size, c='b', alpha=alpha, label='Versicolor')
    plt.scatter(df[df.iris=='Iris-virginica'].sepal_len, df[df.iris=='Iris-virginica'].sepal_wid, s=size, c='g', alpha=alpha, label='Virginica')
    plt.xlabel('Sepal length (cm)')
    plt.ylabel('Sepal width (cm)')
    plt.legend(loc='upper right')
    plt.show()
# plot_scatter()

def k_nn(point, k=3, df_pred=0, df_classes=0):
    '''get the k-closest neighbors from the point and outputs the majority class
      df_pred = predictor(s); df_classes = classes (len(df_pred) = len(df_classes))
      point is a dictionary whose keys are the same as column names of df_pred'''
    temp = df_pred.copy()
    for feat in df_pred.columns:
        temp[feat] = (temp[feat] - point[feat])**2
    distance = np.sqrt(temp.sum(axis=1))
    distance.sort_values(inplace=True, ascending=True)
    return df_classes.iloc[distance[:k].index].value_counts().index[0]
    # Note: value_counts counts the number of occurence of each value in df_classes and sort them in descending order
    

test_idx = np.random.uniform(0, 1, len(df)) <= 0.3
df_train = df[test_idx==False]
df_train.index = range(len(df_train))
df_test = df[test_idx==True]
df_test.index = range(len(df_test))
# df_test['prediction'] = df_test[features].apply(k_nn, axis=1, k=7, df_pred=df_train[features], df_classes=df_train.iris)
# df_test['good_prediction'] = df_test.iris == df_test.prediction
# accuracy = float(len(df_test[df_test.good_prediction==True])) / len(df_test)

def test_accuracy_with_k():
    accuracy = []
    k = []
    for i in xrange(1, 11):
        k.append(i)
        df_test['prediction'] = df_test[features].apply(k_nn, axis=1, k=i, df_pred=df_train[features], df_classes=df_train.iris)
        df_test['good_prediction'] = df_test.iris == df_test.prediction
        accuracy.append(float(len(df_test[df_test.good_prediction==True])) / len(df_test))
        
    fig = plt.figure('K-NN accuracy as a function of k')
    plt.plot(k, accuracy)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.xlim([min(k)-1, max(k)])
    plt.ylim([0,1])
    plt.show()
test_accuracy_with_k()