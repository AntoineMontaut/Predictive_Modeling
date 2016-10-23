'''Unit4 Lesson3: naive Bayes' classification'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('ideal-weight/ideal_weight.csv')
df.columns = [x.replace("'", '') for x in df.columns]
df.rename(columns={'diff': 'difference'}, inplace=True)
df.sex = df.sex.map(lambda x: x.replace("'", ''))
print('Number of missing data: {0}'.format(df.isnull().sum().sum()))
print('Out of {0} entries, the dataset contains {1} women and {2} men'.format(
         len(df), len(df[df.sex=='Female']), len(df[df.sex=='Male'])))
df['sex_cat'] = df.sex.map(lambda x: 0 if x=='Female' else 1)
# print(df.info())

def plot_hists():
    fig = plt.figure('Ideal and Actual Weights', figsize=(10,8))
    
    plt.subplot(2,1,1)
    plt.hist(df.actual, alpha=.5, label='Actual')
    plt.hist(df.ideal, alpha=.5, label='Ideal')
    plt.legend(loc='upper right')
    plt.xlabel('Weight (lbs)')
    
    plt.subplot(2,1,2)
    plt.hist(df.difference, alpha=.5, label='Diff.', color='r')
    plt.xlabel('Weight Difference (lbs)')
    plt.legend(loc='upper right')
    
    plt.show()

# plot_hists()

'''NAIVE BAYES CLASSIFIER'''
clf = GaussianNB()
clf.fit(df[['actual', 'ideal', 'difference']], df.sex)
score_train = clf.score(df[['actual', 'ideal', 'difference']], df.sex)
print('Naive Bayes classifier score on training data: {0}'.format(round(score_train, 3)))
print('Out of {0} entries, {1} were correctly labeled (i.e. {2} were mislabeled)'.format(
         len(df), round(len(df)*score_train, 0), len(df) - round(len(df)*score_train, 0)))
         
'''PREDICTIONS'''
pred = {'actual': [145, 160],
             'ideal': [160, 145],
             'diff': [-15, 15]}
print('\nPredictions:')
for i in xrange(len(pred['actual'])):
    prediction = clf.predict([[pred[x][i] for x in ['actual', 'ideal', 'diff']]])
    print(" -Given an actual and ideal weights of {0} and {1}, respectively, and a weight \
difference of {2}, the naive Bayes' classifier predict that this person is a {3}".format(
    pred['actual'][i], pred['ideal'][i], pred['diff'][i], str(prediction[0]).lower()))