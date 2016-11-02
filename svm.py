'''Unit4 Lesson6'''

from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib.colors import ListedColormap
import numpy as np

iris = datasets.load_iris()
# print(iris['target_names'])
# print(iris['data'][:5, :])
# print(iris.keys())
# print(iris['feature_names'])
# print(iris['DESCR'])

df = pd.DataFrame(iris['data'], columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid'])
df['target'] = iris['target']
df['iris'] = df.target.map(lambda x: iris['target_names'][x])
# print(df.head())
# print(df.tail())
print(df.info())

def plot_data():
    
    fig = plt.figure('Iris', figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.scatter(df[df.iris=='setosa'].sepal_wid, df[df.iris=='setosa'].petal_len, c='b', label='Setosa')
    plt.scatter(df[df.iris=='versicolor'].sepal_wid, df[df.iris=='versicolor'].petal_len, c='r', label='Versicolor')
    plt.scatter(df[df.iris=='virginica'].sepal_wid, df[df.iris=='virginica'].petal_len, c='g', label='Virginica')
    plt.legend(loc='upper right')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Petal Length (cm)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df[df.iris=='setosa'].sepal_wid, df[df.iris=='setosa'].petal_len, c='b', label='Setosa')
    plt.scatter(df[df.iris=='versicolor'].sepal_wid, df[df.iris=='versicolor'].petal_len, c='r', label='Versicolor')
    plt.legend(loc='upper right')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Petal Length (cm)')
    
    
    
    # plt.scatter(iris.data[:100, 1], iris.data[:100, 2], c=iris.target[:100])
    # plt.xlabel(iris.feature_names[1])
    # plt.ylabel(iris.feature_names[2])

    # plt.show()
    
# plot_data()

X = iris.data[0:100, :]
y = iris.target[0:100]

svc = svm.SVC(kernel='linear')
# svc.fit(X, y)

def plot_estimator(estimator, X, y, title):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(title)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
    # plt.show()

plot_estimator(svc, X, y, 'Setosa & Versicolor')

X = iris.data[50:, :]
y = iris.target[50:]
plot_estimator(svc, X, y, 'Versicolor & Virginica')

plt.show()
