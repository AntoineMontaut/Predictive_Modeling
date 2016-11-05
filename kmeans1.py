'''Unit4 Lesson5: Clustering'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

df = pd.read_csv('iris/iris.data.csv')
df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'iris']
features = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
features_long = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']

def matrix_scatter_plot():
    fig = plt.figure('Features Matrix Plot for Iris Setosa, Versicolor, and Virginica', figsize=(20,10))
    (alpha, size) = (.7, 50)
    for i in xrange(4):
        for j in xrange(4):
            if i != j:
                plt.subplot(4, 4, i*4+j+1)
                plt.scatter(df[df.iris=='Iris-setosa'][features[i]], df[df.iris=='Iris-setosa'][features[j]], s=size, c='r', alpha=alpha, label='Set.')
                plt.scatter(df[df.iris=='Iris-versicolor'][features[i]], df[df.iris=='Iris-versicolor'][features[j]], s=size, c='b', alpha=alpha, label='Ver.')
                plt.scatter(df[df.iris=='Iris-virginica'][features[i]], df[df.iris=='Iris-virginica'][features[j]], s=size, c='g', alpha=alpha, label='Vir.')
                plt.xlabel(features_long[j])
                plt.ylabel(features_long[i])
                if i == 0 and j == 3:
                    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
matrix_scatter_plot()

