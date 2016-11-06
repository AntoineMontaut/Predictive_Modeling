'''Apply PCA to UN dataset from lesson 5'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import cluster

df_UN_0 = pd.read_csv('un/un.csv')
df_UN = df_UN_0[['lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita']]
df_UN = df_UN.dropna()
features = cluster.vq.whiten(df_UN[['lifeMale', 'lifeFemale', 'infantMortality']])
# print(df_UN.info())
# print('Number of countries in the dataset: {0}'.format(len(df.country.unique())))

x_columns = list(df_UN.columns)
# [x_columns.remove(elem) for elem in ['GDPperCapita', 'country', 'region']]
x_columns = ['lifeMale', 'lifeFemale', 'infantMortality']

df_UN_X = df_UN[x_columns]
df_UN_X_std = pd.DataFrame(StandardScaler().fit_transform(df_UN_X), columns=x_columns)
df_UN_y = pd.Series(df_UN['GDPperCapita'])

centroid, label = cluster.vq.kmeans2(features, 3)
df_UN['label'] = label

def plot_explained_variance_ratio_UN():
    pca_max_comp = PCA(n_components=len(df_UN_X_std.columns))
    pca_max_comp.fit(df_UN_X_std)
    n_components = [x + 1 for x in range(len(df_UN_X_std.columns))]
    
    plt.figure('PCA\'s Explained Variance Ratio Using Max Number of Components', figsize=(8, 5))
    plt.plot(n_components, pca_max_comp.explained_variance_ratio_, marker='o', c='r')
    plt.xlabel('Component #')
    plt.ylabel('Explained Variance Ratio')
    # plt.xlim((.5, 4.5))
    plt.xticks(range(1, 12))
    plt.tight_layout()
    plt.show()
    
# plot_explained_variance_ratio_UN()

obs = ['lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita']

def plot_UN():
    fig = plt.figure('lifeMale, lifeFemale, and infantMortality vs. GDPperCapita, colored by cluster')
    color = ['b', 'r', 'g']
    i=0
    for ob in obs[:-1]:
        i += 1
        plt.subplot(2, 2, i)
        for lab in xrange(3):
            plt.scatter(df_UN[df_UN.label==lab].GDPperCapita, 
                           df_UN[df_UN.label==lab][ob], c=color[lab])
        plt.xlim([0,50000])
        plt.xlabel('GDP per Capita')
        plt.ylabel(ob)

    plt.tight_layout()

plot_UN()
    
def plot_UN_PCA():
    pca_UN = PCA(n_components=3)
    pca_UN.fit(df_UN_X_std)
    df_UN_X_pca = pd.DataFrame(pca_UN.transform(df_UN_X_std))
    centroid_pca, label_pca = cluster.vq.kmeans2(df_UN_X_pca[[0, 1, 2]], 3)
    df_UN['label_pca'] = label_pca
    label_pca = pd.Series(label_pca)
    
    fig = plt.figure('First 3 PCA components vs. GDPperCapita, colored by cluster\n(Clustering AFTER PCA)')
    color = ['b', 'r', 'g']
    i=0
    for ob in xrange(3):
        i += 1
        plt.subplot(2, 2, i)
        for lab in xrange(3):
            plt.scatter(df_UN[df_UN.label_pca==lab].GDPperCapita, 
                           df_UN_X_pca[label_pca==lab][ob], c=color[lab])
        plt.xlim([0,50000])
        plt.xlabel('GDP per Capita')
        plt.ylabel('Component #{0}'.format(ob+1))

    plt.tight_layout()
    
    fig = plt.figure('First 3 PCA components vs. GDPperCapita, colored by cluster\n(Clustering BEFORE PCA)')
    color = ['b', 'r', 'g']
    i=0
    for ob in xrange(3):
        i += 1
        plt.subplot(2, 2, i)
        for lab in xrange(3):
            plt.scatter(df_UN[df_UN.label==lab].GDPperCapita, 
                           df_UN_X_pca[label==lab][ob], c=color[lab])
        plt.xlim([0,50000])
        plt.xlabel('GDP per Capita')
        plt.ylabel('Component #{0}'.format(ob+1))

    plt.tight_layout()
    
plot_UN_PCA()

def plot_UN_LDA():
    lda_UN = LinearDiscriminantAnalysis(n_components=3)
    lda_UN.fit(df_UN_X_std, df_UN.label)
    df_UN_X_lda = pd.DataFrame(lda_UN.transform(df_UN_X_std))
    centroid_lda, label_lda = cluster.vq.kmeans2(df_UN_X_lda[[0, 1]], 3)
    df_UN['label_lda'] = label_lda
    label_lda = pd.Series(label_lda)
    
    fig = plt.figure('First 3 LDA components vs. GDPperCapita, colored by cluster')
    color = ['b', 'r', 'g']
    i=0
    for ob in xrange(2):
        i += 1
        plt.subplot(1, 2, i)
        for lab in xrange(3):
            plt.scatter(df_UN[df_UN.label==lab].GDPperCapita, 
                           df_UN_X_lda[label==lab][ob], c=color[lab])
        plt.xlim([0,50000])
        plt.xlabel('GDP per Capita')
        plt.ylabel('Component #{0}'.format(ob+1))

    plt.tight_layout()
    
plot_UN_LDA()

plt.show()