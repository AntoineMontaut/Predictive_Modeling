'''Unit4 Lesson8: cross-validation'''

import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn import metrics as skmetrics
import numpy as np

iris = datasets.load_iris()
features = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']

df = pd.DataFrame(iris['data'], columns=features)
df['target'] = iris['target']
df['iris'] = df.target.map(lambda x: iris['target_names'][x])

svc = svm.SVC()

accuracy = []
f1 = []
precision = []
recall = []

kf = KFold(n_splits=5)
for train, test in kf.split(df.target):
    df_train = df.loc[train, :]
    df_train.index = range(len(df_train))
    df_test = df.loc[test, :]
    df_test.index = range(len(df_test))
    
    svc.fit(df_train[features], df_train.iris)
    df_test['prediction'] = svc.predict(df_test[features])
    
    accuracy.append(skmetrics.accuracy_score(df_test.iris, df_test.prediction))
    f1.append(skmetrics.f1_score(df_test.iris, df_test.prediction, average='weighted'))
    precision.append(skmetrics.precision_score(df_test.iris, df_test.prediction, average='weighted'))
    recall.append(skmetrics.recall_score(df_test.iris, df_test.prediction, average='weighted'))
    
print('\nAverage scores in a 5-fold cross-validation of the svc model on the iris dataset:\
\n\t- Accuracy: {0}\n\t- F1 score: {1}\n\t- Precision: {2}\n\t- Recall: {3}'.format(
round(np.mean(accuracy), 3), round(np.mean(f1), 3), round(np.mean(precision), 3), round(np.mean(recall), 3)))