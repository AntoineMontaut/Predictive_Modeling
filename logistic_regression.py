'''
Unit3 Lesson4: Logistic Regression
'''

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

ds_loans = pd.read_csv("loansData_clean.csv")
# print(ds_loans.head(5))
# print(ds_loans.info())
ds_loans = ds_loans.dropna()

#IT_TF=0 when IR<12%, IR_TF = 1 when IR>12%
ds_loans["IR_TF"] = ds_loans["Interest.Rate"].map(lambda x: 0 if x>.12 else 1)
# print(ds_loans[["Interest.Rate", "IR_TF"]].head(5))
# print(ds_loans[["Interest.Rate", "IR_TF"]][ds_loans["Interest.Rate"]>0.12].head(5))
# print(ds_loans[["Interest.Rate", "IR_TF"]][ds_loans["Interest.Rate"]<0.12].head(5))

#statsmodels need an intercept column in the DataFrame
ds_loans["Intercept"] = 1.0

#create a list of all independent variables
ind_vars = ["Intercept", "FICO.Score", "Amount.Requested"]
# ind_vars = []
# [ind_vars.append(col) for col in ds_loans.columns]

'''
prob of getting a loan for $10,000 at an interest rate < 12% with a FICO score of 750
'''
model = sm.Logit(ds_loans["IR_TF"], ds_loans[ind_vars]).fit()
# print(model.summary())
(cst, coef_fico, coef_amount) = [model.params[i] for i in xrange(3)]

def logistic_function(fico, amount):
    '''Outputs probability that loan will be given at interest rate < 12%'''
    #prob = 1 / (1 + np.exp(-mx - b))
    return 1.0 / (1.0 + np.exp(-coef_fico*fico - coef_amount*amount - cst))
    
def print_proba(fico, amount):
    '''Print the probability to have a loan with interest rate < 12% depending on FICO and loan amount'''
    print("\nThe probability to have a loan with interest rate less than 12% with a FICO score of {0} and a loan amount of ${1} is {2}%.".format\
    (fico, amount, round(logistic_function(fico, amount),3)*100))
    
print_proba(720, 10000)

#plot probability to get a loan with interest rate < 12% for different loan amounts as a function of FICO score
def plot_proba():
    fico_scores = np.linspace(ds_loans["FICO.Score"].min(), ds_loans["FICO.Score"].max(), 100)
    probs_5k = [logistic_function(fico_temp, 5000) for fico_temp in fico_scores]
    probs_10k = [logistic_function(fico_temp, 10000) for fico_temp in fico_scores]
    probs_20k = [logistic_function(fico_temp, 20000) for fico_temp in fico_scores]
    plt.plot(fico_scores, probs_5k, label="Loan amount = $5,000")
    plt.plot(fico_scores, probs_10k, label="Loan amount = $10,000")
    plt.plot(fico_scores, probs_20k, label="Loan amount = $20,000")
    plt.legend(loc="lower right")
    plt.xlabel("FICO score")
    plt.ylabel("Probability to  get the loan")
    plt.title("Probability to get a loan with interest rate < 12%\n for different loan amounts as a function of FICO score")
    plt.show()
    
plot_proba()

def get_loan(fico, amount):
    prob_to_get_loan = logistic_function(fico, amount)
    if prob_to_get_loan >= 0.7:
        print("\nGiven your FICO score of {0} and for an amount of ${1}, you will get a loan with interest rate of less than 12%.".format\
        (fico, amount))
    else:
        print(("\nGiven your FICO score of {0} and for an amount of ${1}, you will NOT get a loan with interest rate of less than 12%.".format\
        (fico, amount)))
        
[get_loan(fico, amount) for (fico, amount) in [(680, 5000), (710, 10000), (750, 20000)]]