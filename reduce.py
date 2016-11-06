'''Unit4 Lesson7: Dimensionality Reduction'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import cluster


iris = datasets.load_iris()
X = iris.data
y = iris.target

features = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
target_names = iris.target_names
df_X = pd.DataFrame(iris.data, columns=features)
df_X_std = pd.DataFrame(StandardScaler().fit_transform(df_X), columns=features)
df_y = pd.Series(iris.target)

# PCA
pca = PCA(n_components=2)
pca.fit(df_X_std)


# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(df_X_std, df_y)

# Variance ratio for PCA and LDA
# print(pca.explained_variance_ratio_)
# print(lda.explained_variance_ratio_)

# Transform X
df_X_pca = pd.DataFrame(pca.transform(df_X_std))
df_X_lda = pd.DataFrame(lda.transform(df_X_std))

# Plot the two transformed X

def plot_pca_lda():
    colors = ['navy', 'turquoise', 'darkorange']
    fig = plt.figure('PCA and LDA (2 components) applied to iris data', figsize=(13,11))
    
    plt.subplot(3, 2, 1)
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(df_X_std[df_y==i]['petal_len'], df_X_std[df_y==i]['petal_wid'], color=color, label=target_name)
    plt.legend(loc='lower right', shadow=False, scatterpoints=1)
    plt.title('Best Two Attributes From Cluster Analysis')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    
    plt.subplot(3, 2, 3)
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(df_X_pca[df_y==i][0], df_X_pca[df_y==i][1], color=color, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA')
    
    plt.subplot(3, 2, 4)
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(df_X_lda[df_y==i][0], df_X_lda[df_y==i][1], color=color, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA')
    
    
    colors = {'0': 'r', '1': 'b', '2': 'g'}
    plt.subplot(3, 2, 5)
    centroid, label_pca = cluster.vq.kmeans2(df_X_pca, 3)
    label_pca = pd.Series(label_pca)
    for i, target_name in zip([0, 1, 2], target_names):
        plt.scatter(df_X_pca[label_pca==i][0], df_X_pca[label_pca==i][1], color=colors[str(i)], label=target_name)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Clustering after PCA')
    
    colors = {'0': 'r', '1': 'b', '2': 'g'}
    plt.subplot(3, 2, 6)
    centroid, label_lda = cluster.vq.kmeans2(df_X_lda, 3)
    label_lda = pd.Series(label_lda)
    for i, target_name in zip([0, 1, 2], target_names):
        plt.scatter(df_X_lda[label_lda==i][0], df_X_lda[label_lda==i][1], color=colors[str(i)], label=target_name)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Clustering after LDA')
    
    plt.tight_layout()
    
plot_pca_lda()

def plot_explained_variance_ratio():
    pca_max_comp = PCA(n_components=len(df_X_std.columns))
    pca_max_comp.fit(df_X_std)
    n_components = [x + 1 for x in range(len(df_X_std.columns))]
    
    plt.figure('PCA\'s Explained Variance Ratio Using Max Number of Components', figsize=(8, 5))
    plt.plot(n_components, pca_max_comp.explained_variance_ratio_, marker='o', c='r')
    plt.xlabel('Component #')
    plt.ylabel('Explained Variance Ratio')
    plt.xlim((.5, 4.5))
    plt.xticks((1, 2, 3, 4))
    plt.tight_layout()
    
# plot_explained_variance_ratio()

plt.show()