'''Unit4 Lesson7: Dimensionality Reduction'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


iris = datasets.load_iris()
X = iris.data
y = iris.target

features = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
target_names = iris.target_names
df_X = pd.DataFrame(iris.data, columns=features)
# df_y = pd.DataFrame(iris.target, columns=['target'])
df_y = pd.Series(iris.target)

# PCA
pca = PCA(n_components=2)
pca.fit(df_X)

# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(df_X, df_y)

# Transform X
df_X_pca = pd.DataFrame(pca.transform(df_X))
df_X_lda = pd.DataFrame(lda.transform(df_X))

# Plot the two transformed X

def plot_pca_lda():
    colors = ['navy', 'turquoise', 'darkorange']
    fig = plt.figure('PCA and LDA (2 components) applied to iris data', figsize=(13,9))
    
    plt.subplot(2, 2, 1)
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(df_X[df_y==i]['petal_len'], df_X[df_y==i]['petal_wid'], color=color, label=target_name)
    plt.legend(loc='lower right', shadow=False, scatterpoints=1)
    plt.title('Best Two Attributes From Cluster Analysis')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    
    plt.subplot(2, 2, 3)
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(df_X_pca[df_y==i][0], df_X_pca[df_y==i][1], color=color, label=target_name)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA')
    
    plt.subplot(2, 2, 4)
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(df_X_lda[df_y==i][0], df_X_lda[df_y==i][1], color=color, label=target_name)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA')
    
    plt.tight_layout()
    
plot_pca_lda()

plt.show()