'''Unit4 Lesson5: Clustering'''

import pandas as pd
import matplotlib.pyplot as plt
from scipy import cluster

df = pd.read_csv('un/un.csv')
print(df.info())
# print('Number of countries in the dataset: {0}'.format(len(df.country.unique())))

'''We want to see how lifeMale, lifeFemale and infantMortality cluster according to GDPperCapita'''
obs = ['lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita']
observations = df[obs].dropna()
features = cluster.vq.whiten(observations[['lifeMale', 'lifeFemale', 'infantMortality']])

distorsions = []
for k in xrange(1, 11):
    codebook, distorsion = cluster.vq.kmeans(features, k)
    distorsions.append(distorsion)
        
fig = plt.figure('lifeMale, lifeFemale, and infantMortality clustering distortion vs. # of centroids')
plt.plot(range(1,11), distorsions, '-o')
plt.xlabel('k')
plt.xlim([1,10])
plt.ylabel('Distorsion')
plt.title('Distorsion vs. k (# of centroids)')

'''We observe that 3 is a good number for k'''

centroid, label = cluster.vq.kmeans2(features, 3)
observations['label'] = label

fig = plt.figure('lifeMale, lifeFemale, and infantMortality vs. GDPperCapita, colored by cluster')
color = ['b', 'r', 'g']
i=0
for ob in obs[:-1]:
    i += 1
    plt.subplot(2, 2, i)
    for lab in xrange(3):
        plt.scatter(observations[observations.label==lab].GDPperCapita, 
                       observations[observations.label==lab][ob], c=color[lab])
    plt.xlabel('GDP per Capita')
    plt.ylabel(ob)

plt.tight_layout()
plt.show()