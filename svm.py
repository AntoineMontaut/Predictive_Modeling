'''Unit4 Lesson6'''

from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import numpy as np
import itertools
from matplotlib import gridspec

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
# print(iris['target_names'])
# print(iris['data'][:5, :])
# print(iris.keys())
# print(iris['feature_names'])
# print(iris['DESCR'])

features = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']

df = pd.DataFrame(iris['data'], columns=features)
df['target'] = iris['target']
df['iris'] = df.target.map(lambda x: iris['target_names'][x])
# print(df.head())
# print(df.tail())
# print(df.info())

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

svc = svm.SVC(kernel='linear', C=1)
# svc.fit(X, y)

def plot_estimator(estimator, X, y, title):

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

# plot_estimator(svc, X, y, 'Setosa & Versicolor')

# X = iris.data[50:, :]
# y = iris.target[50:]
# plot_estimator(svc, X, y, 'Versicolor & Virginica')

df1 = df[df.iris != 'virginica']
df2 = df[df.iris != 'versicolor']
df3 = df[df.iris != 'setosa']
combinations_iter = itertools.combinations(features, 2)
combinations = []
for com in combinations_iter:
    combinations.append(com)

def plot_all():
    for df_sub in [df1, df2, df3]:
        fig = plt.figure(' and '.join(df_sub.iris.unique()), figsize=(13, 8))
        gs = gridspec.GridSpec(2, 3)
        
        i = -1
        for combination in combinations:
            i += 1
            axes = plt.subplot(gs[i/3, i%3])
            
            svc.fit(df_sub[list(combination)], df_sub['target'])
            x_min, x_max = df_sub[combination[0]].min() - .1, df_sub[combination[0]].max() + .1
            y_min, y_max = df_sub[combination[1]].min() - .1, df_sub[combination[1]].max() + .1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            axes.pcolormesh(xx, yy, Z, cmap=cmap_light)

            # Plot also the training points
            plt.scatter(df_sub[combination[0]], df_sub[combination[1]], c=df_sub['target'], cmap=cmap_bold)
            # axes.scatter(df_sub[combination[0]], df_sub[combination[1]], c=df_sub['target'], cmap=cmap_bold)
            
            plt.axis('tight')
            # plt.axis('off')
            plt.xlabel(combination[0])
            plt.ylabel(combination[1])
        
# plot_all()

def svc_three_flowers():
    fig = plt.figure('Three Flowers', figsize=(13, 8))
    gs = gridspec.GridSpec(2, 3)
    i = -1
    for combination in combinations:
        i += 1
        axes = plt.subplot(gs[i/3, i%3])
            
        svc.fit(df[list(combination)], df['target'])
        x_min, x_max = df[combination[0]].min() - .1, df[combination[0]].max() + .1
        y_min, y_max = df[combination[1]].min() - .1, df[combination[1]].max() + .1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        axes.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(df[combination[0]], df[combination[1]], c=df['target'], cmap=cmap_bold)
        # axes.scatter(df[combination[0]], df[combination[1]], c=df['target'], cmap=cmap_bold)
        
        plt.axis('tight')
        # plt.axis('off')
        plt.xlabel(combination[0])
        plt.ylabel(combination[1])
        
# svc_three_flowers()

plt.show()


'''---------------------USE SVM TO PREDICT FLOWER TYPE---------------------'''


# print(df.head())
df_train, df_test = train_test_split(df, test_size=0.4)
# df_train.index = range(len(df_train))
# df_test.index = range(len(df_test))

svc = svm.SVC(kernel='linear', C=1)
svc.fit(df_train[features], df_train['iris'])

df_test['prediction_svc'] = svc.predict(df_test[features])
score_svc = svc.score(df_test[features], df_test['iris'])

print('\nUsing 60% of the data for training, we get a score of {0} using the svc model (C=1).'.format(
         round(score_svc, 3)))
         
# print(df_train.head())
# print(df_test.head())
         
# See the impact of C (margin)
# scores_margin = {}
# for margin in xrange(2, 6):
    # svc = svm.SVC(kernel='linear', C=margin).fit(df_train[features], df_train['iris'])
    # scores_margin[str(margin)] = svc.score(df_test[features], df_test['iris'])
    
# print('\nImpact of margin parameter in score:')
# for margin in xrange(2,6):
    # print('\tC = {0}: score = {1}'.format(margin, 
             # round(scores_margin[str(margin)], 3)))
             