'''Unit4 Lesson2'''

import pandas as pd
import matplotlib.pyplot as plt

activity_list = ['Walking', 'Walking upstairs', 'Walking downstairs',
                      'Sitting', 'Standing', 'Laying']
activity_dict = {'1': 'Walking', '2': 'Walking upstairs', '3': 'Walking downstairs',
                       '4': 'Sitting', '5': 'Standing', '6': 'Laying'}

open_features = open('./UCI HAR Dataset/features.txt', 'r')
features = []
for line in open_features.readlines():
    dummy, feature = str(line).rstrip().split(' ')
    for char in ['-', '(', ',']:
        feature = feature.replace(char, '_')
    feature = feature.replace('()', '')
    feature = feature.replace(')', '')
    feature = feature.replace('__', '_')
    for elem in ['BodyBody', 'Body', 'Mag']:
        feature = feature.replace(elem, '')
    if feature[-1] == '_':
        feature = feature[:-1]
    features.append(feature)
    
# print('# of features: {0}'.format(len(features)))
# print(features)

# features_sorted = list(features)
# features_sorted.sort()
# for feature in features_sorted:
    # print(feature)
    
X_train =pd.read_table('./UCI HAR Dataset/train/X_train.txt', sep ='\s+', names=features)
# print(X_train.info())
open_y_train = open('./UCI HAR Dataset/train/y_train.txt')
y_train = []
for line in open_y_train.readlines():
    y_train.append(str(line).rstrip()[0])
y_train = pd.DataFrame(y_train, columns=['activity_cat'])
y_train['activity'] = y_train.activity_cat.map(lambda x: activity_dict[x])
X_train['activity'] = y_train['activity']
X_train['activity_cat'] = y_train['activity_cat']
# print(y_train.info())
# print('# of data points: {0}'.format(len(y_train)))

# plt.hist(X_train[X_train.activity=='Walking'].fAcc_mean)
# plt.show()
fig = plt.figure('Mean Body Acceleration')
i = 0
for act in activity_list:
    i += 1
    plt.subplot(2,3,i)
    plt.hist(X_train[X_train.activity==act].fAcc_mean)
    plt.xlim([-1.0, 1.0])
    plt.title(act)
    if i == 1:
        plt.ylabel('Dynamic Activity')
    if i == 4:
        plt.ylabel('Static Activity')
plt.show()