'''
Unit2 Lesson5: Multivariate Analysis of Loan Data
'''

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("loansData_clean.csv")
df = df.dropna()
print(df.info())
# print(pd.Series.unique(df["Home.Ownership"]))
df["Home.Ownership.Ord"] = pd.Categorical(df["Home.Ownership"]).labels
df["Annual.Income"] = df["Monthly.Income"] * 12
# df["Home.Ownership.Ord"] = df["Home.Ownership.Ord"].map(lambda x: 4 if x ==0 else x)
# print(df.head())

# fig = plt.figure("Interest rate")
# plt.scatter(df["Annual.Income"], df["Interest.Rate"])
# plt.show()
# plt.scatter(df["Annual.Income"], df["Home.Ownership.Ord"])
# plt.show()

x = df["Annual.Income"]
X = sm.add_constant(x)
print(x.head())
model_1 = sm.OLS(df["Interest.Rate"], X).fit()
print(model_1.summary())

x = df[["Annual.Income", "Home.Ownership.Ord"]]
X = sm.add_constant(x)
model_2 = sm.OLS(df["Interest.Rate"], X).fit()
print(model_2.summary())

x = df[["Annual.Income", "Home.Ownership.Ord"]]
x["Income.Ownership.Interaction"] = df["Annual.Income"] * df["Home.Ownership.Ord"]
X = sm.add_constant(x)
model_3 = sm.OLS(df["Interest.Rate"], X).fit()
print(model_3.summary())