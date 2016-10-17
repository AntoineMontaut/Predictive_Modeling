'''Unit4 Lesson1'''

import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import sklearn.metrics as skmet

np.random.seed(414)

X = np.linspace(0, 15, 1000)
y = 3*np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

linear_fit = smf.ols(formula='y ~ 1 + X', data=train_df).fit()
quadratic_fit = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()

# print(linear_fit.summary())
# print(linear_fit.params)
linear_fit_training = linear_fit.params[0] + train_X*linear_fit.params[1]
linear_fit_test = linear_fit.params[0] + test_X*linear_fit.params[1]

# print(quadratic_fit.summary())
# print(quadratic_fit.params)
quadratic_fit_training = quadratic_fit.params[0] + quadratic_fit.params[1]*train_X + quadratic_fit.params[2]*train_X**2
quadratic_fit_test = quadratic_fit.params[0] + quadratic_fit.params[1]*test_X +  + quadratic_fit.params[2]*test_X**2

print('Mean squared error on training data is:\n\t{0} for linear fit\n\t{1} for quadratic fit'.format(
         round(skmet.mean_squared_error(linear_fit_training, train_y), 3), 
         round(skmet.mean_squared_error(quadratic_fit_training, train_y), 3)))

print('Mean squared error on training data is:\n\t{0} for linear fit\n\t{1} for quadratic fit'.format(
         round(skmet.mean_squared_error(linear_fit_test, test_y), 3), 
         round(skmet.mean_squared_error(quadratic_fit_test, test_y), 3)))