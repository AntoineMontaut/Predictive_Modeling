'''Unit4 Lesson1'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
import sklearn.metrics as skmet

df = pd.read_csv("loansData_clean.csv")
df = df.dropna()
print(df.info())
df["Annual.Income"] = df["Monthly.Income"] * 12

X = sm.add_constant(df['Amount.Requested'])
base_model = sm.OLS(df['Interest.Rate'], X).fit()
print('BASE MODEL:')
print(base_model.summary())



mean_abs_error = []
mean_sq_error = []
r_squared = []

i = 0
kf = KFold(n_splits=10)
for train, test in kf.split(df['Interest.Rate']):
    Y_train = [df['Interest.Rate'][index] for index in train]
    Y_test = [df['Interest.Rate'][index] for index in test]
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    x_train = [df['Amount.Requested'][index] for index in train]
    x_test = [df['Amount.Requested'][index] for index in test]
    X_train = np.array(x_train)
    X_train = sm.add_constant(X_train)
    
    model = sm.OLS(Y_train, X_train).fit()
    y_pred = [model.params[0] + x*model.params[1] for x in x_test]
    Y_pred = np.array(y_pred)
    
    mae = skmet.mean_absolute_error(Y_test, Y_pred)
    mean_abs_error.append(mae)
    mse = skmet.mean_squared_error(Y_test, Y_pred)
    mean_sq_error.append(mse)
    r2 = skmet.r2_score(Y_test, Y_pred)
    r_squared.append(r2)
    
    i += 1
    print('Fold #{0}'.format(i))
    rounded = [round(x, 3) for x in [mae, mse, r2]]
    print('MAE = {0}, MSE = {1}, R2 = {2}'.format(rounded[0], rounded[1], rounded[2]))

mean_abs_error = np.array(mean_abs_error)
mean_sq_error = np.array(mean_sq_error)
r_squared = np.array(r_squared)
scores = [mean_abs_error.mean(), mean_sq_error.mean(), r_squared.mean()]
rounded_scores = [round(x, 3) for x  in scores]
print('\nOverall score:\nMAE = {0}, MSE = {1}, R2 = {2}'.format(
         rounded_scores[0], rounded_scores[1], rounded_scores[2]))

# test = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
# kf = KFold(n_splits=3)
# i = 0
# for train, test in kf.split(test):
    # i += 1
    # print('Fold #{0}'.format(i))
    # print(train, test)