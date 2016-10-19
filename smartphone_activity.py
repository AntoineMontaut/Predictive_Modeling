'''Unit4 Lesson2'''

import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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
# X_train['activity'] = y_train['activity']
# X_train['activity_cat'] = y_train['activity_cat']
# print(y_train.info())
# print('# of data points: {0}'.format(len(y_train)))

# plt.hist(X_train[X_train.activity=='Walking'].fAcc_mean)
# plt.show()
# fig = plt.figure('Mean Body Acceleration')
# i = 0
# for act in activity_list:
    # i += 1
    # plt.subplot(2,3,i)
    # plt.hist(X_train[X_train.activity==act].fAcc_mean)
    # plt.xlim([-1.0, 1.0])
    # plt.title(act)
    # if i == 1:
        # plt.ylabel('Dynamic Activity')
    # if i == 4:
        # plt.ylabel('Static Activity')
# plt.show()

# X_train = X_train.select(lambda x: not re.search('.bands.', x), axis=1)
# X_train = X_train.filter(like)
to_keep = []
# flags_out = ['angle', 'band', 'Mag']
flags_out = ['angle', 'band', 'X', 'Y', 'Z', 'arCoeff']
flags_in = ['mean', 'std', 'skewness', 'kurtosis']
for col in X_train.columns:
    reject = False
    for flag in flags_out:
        if re.search(flag, col):
            reject = True
            break
    if not reject:
        reject = True
        for flag in flags_in:
            if re.search(flag, col):
                reject = False
                break
    if not reject and (col not in to_keep):
        to_keep.append(col)

to_keep.sort()
# for stuff in to_keep:
    # print stuff
X_train = X_train[to_keep]
# print(X_train.info())

clf = RandomForestClassifier(n_estimators=50)
clf = clf.fit(X_train, y_train.activity)
scores = cross_val_score(clf, X_train, y_train.activity)
print('Random Forest mean score on training data: {0}'.format(round(scores.mean(), 3)))
# print(clf.feature_importances_)

feat_imp = pd.Series(clf.feature_importances_, index=X_train.columns)
print(feat_imp.sort_values(ascending=False))
# i = 0
# # print('Features importance:')
# for col in X_train.columns:
    # # print('\t-{0}: {1}'.format(col, round(feat_imp[i], 3)))
    # i += 1