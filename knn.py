'''Unit4 Lesson4'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

df = pd.read_csv('iris/iris.data.csv')
df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'iris']
features = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
# print(df.info())
# print(df['iris'].unique())
# test_idx = np.random.uniform(0, 1, len(df)) <= 0.3

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

random_point = {}
i = 0
for feat in features:
    random_point[feat] = random.uniform(df[feat].min(), df[feat].max())
    i += 1

def k_nn(k, point, df_in, df_out):
    '''get the k-closest neighbors from the point and outputs the majority class
      df_in = predictor(s); df_out = target variable
      point is a dictionary whose keys are the same as column names of df_in'''
    temp = df_in.copy()
    for feat in df_in.columns:
        temp[feat] = (temp[feat] - point[feat])**2
    distance = np.sqrt(temp.sum(axis=1))
    distance.sort_values(inplace=True, ascending=False)
    return df_out.iloc[distance[:k].index].value_counts().index[0]
    # Note: value_counts counts the number of occurence of each value in df_out and sort them in descending order
    
    
# k_nn(3, random_point, df[features], df.iris)


