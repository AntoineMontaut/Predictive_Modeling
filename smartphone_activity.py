'''Unit4 Lesson2'''

import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

activity_list = ['Walking', 'Walking upstairs', 'Walking downstairs',
                      'Sitting', 'Standing', 'Laying']
activity_dict = {'1': 'Walking', '2': 'Walking upstairs', '3': 'Walking downstairs',
                       '4': 'Sitting', '5': 'Standing', '6': 'Laying'}


'''IMPORT FEATURES LIST AND TRIM NAMES'''
open_features = open('./UCI HAR Dataset/features.txt', 'r')
features = []
for line in open_features.readlines():
    dummy, feature = str(line).rstrip().split(' ')
    for char in ['-', '(', ',']:
        feature = feature.replace(char, '_')
    feature = feature.replace('()', '')
    feature = feature.replace(')', '')
    feature = feature.replace('__', '_')
    for elem in ['BodyBody', 'Body']:#, 'Mag']:
        feature = feature.replace(elem, '')
    if feature[-1] == '_':
        feature = feature[:-1]
    features.append(feature)
    
# print('# of features: {0}'.format(len(features)))
# features_sorted = list(features)
# features_sorted.sort()
# for feature in features_sorted:
    # print(feature)
    

'''IMPORT AND CLEAN TRAINING DATA'''
X_train =pd.read_table('./UCI HAR Dataset/train/X_train.txt', sep ='\s+', names=features)
# print(X_train.info())
open_y_train = open('./UCI HAR Dataset/train/y_train.txt')
y_train = []
for line in open_y_train.readlines():
    y_train.append(str(line).rstrip()[0])
y_train = pd.DataFrame(y_train, columns=['activity_cat'])
y_train['activity'] = y_train.activity_cat.map(lambda x: activity_dict[x])
# print(y_train.info())

def preview_plots():
    X_train.activity = y_train.activity
    i = 0
    for act in activity_list:
        i += 1
        plt.subplot(2,3,i)
        plt.hist(X_train[X_train.activity==act].fAccMag_mean)
        plt.xlim([-1.0, 1.0])
        plt.title(act)
        if i == 1:
            plt.ylabel('Dynamic Activity')
        if i == 4:
            plt.ylabel('Static Activity')
    plt.show()
    X_train.drop('activity', axis=1, inplace=True)

# preview_plots()
    
to_keep = []
flags_out = ['angle', 'band', 'arCoeff', 'Mag']
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


'''TRAIN RANDOM FOREST ON TRAINING DATA'''
clf = RandomForestClassifier(n_estimators=500)
clf = clf.fit(X_train, y_train.activity)
score_train = clf.score(X_train, y_train.activity)
print('Random Forest accuracy on training data: {0}'.format(round(score_train, 5)))
print('oob score: {0}',format(round(clf.oob_score_, 3)))
# score_train = cross_val_score(clf, X_train, y_train.activity)
# print('Random Forest mean score on training data: {0}'.format(round(score_train.mean(), 3)))

feat_imp = pd.Series(clf.feature_importances_, index=X_train.columns)
feat_imp_sorted = feat_imp.sort_values(ascending=False)
print('Most important features:')
print(feat_imp_sorted[:10])


'''IMPORT TESTING DATA AND APPLY RANDOM FORST MODEL'''
X_test =pd.read_table('./UCI HAR Dataset/test/X_test.txt', sep ='\s+', names=features)
X_test = X_test[to_keep]
open_y_test = open('./UCI HAR Dataset/test/y_test.txt')
y_test = []
for line in open_y_test.readlines():
    y_test.append(str(line).rstrip()[0])
y_test = pd.DataFrame(y_test, columns=['activity_cat'])
y_test['activity'] = y_test.activity_cat.map(lambda x: activity_dict[x])

prediction = clf.predict(X_test)
score_test = clf.score(X_test, y_test.activity)
precision_test = precision_score(y_test.activity, prediction, average='micro')
recall_test = recall_score(y_test.activity, prediction, average='micro')
scores_rouned = map(lambda x: round(x, 5), [score_test, precision_test, recall_test])
print('\nRandom Forest scores on test data (global):\n\tAccuracy: {0}\n\tPrecision: {1}\n\tRecall: {2}'.format(
         scores_rouned[0], scores_rouned[1], scores_rouned[2]))
precision_test = precision_score(y_test.activity, prediction, average='macro')
recall_test = recall_score(y_test.activity, prediction, average='macro')
scores_rouned = map(lambda x: round(x, 5), [score_test, precision_test, recall_test])
print('\nRandom Forest scores on test data (per label and unweigthed averaged):\n\tAccuracy: {0}\n\tPrecision: {1}\n\tRecall: {2}'.format(
         scores_rouned[0], scores_rouned[1], scores_rouned[2]))
precision_test = precision_score(y_test.activity, prediction, average='weighted')
recall_test = recall_score(y_test.activity, prediction, average='weighted')
scores_rouned = map(lambda x: round(x, 5), [score_test, precision_test, recall_test])
print('\nRandom Forest scores on test data (per label and weigthed averaged):\n\tAccuracy: {0}\n\tPrecision: {1}\n\tRecall: {2}'.format(
         scores_rouned[0], scores_rouned[1], scores_rouned[2]))