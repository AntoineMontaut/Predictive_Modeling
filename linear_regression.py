'''
Unit2 Lesson3: Linear Regression and Correlation
'''

import pandas as pd
import plots
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

ds_loans = pd.read_csv("loansData.csv")
# print(ds_loans.head(5))
print(ds_loans.info())
ds_loans = ds_loans.dropna()

fico = "FICO.Score"
fico_name = fico.replace(".", " ")
int_rate = "Interest.Rate"
int_rate_name = int_rate.replace(".", " ")
length = "Loan.Length"
length_name = length.replace(".", " ")
amt_req = "Amount.Requested"
amt_req_name = amt_req.replace(".", " ")
income = "Monthly.Income"
income_name = income.replace(".", " ")

ds_loans[fico] = ds_loans["FICO.Range"].map(lambda x: int(str(x).rstrip().split("-")[0]))
ds_loans[int_rate] = ds_loans[int_rate].map(lambda x: float(str(x).rstrip()[:-1])/100)
ds_loans[length] = ds_loans[length].map(lambda x: int(str(x).rstrip().split(" ")[0]))

ds_loans.to_csv("loansData_clean.csv", header=True, index=False)

# for col in [fico, int_rate, length]:
    # print(ds_loans[col][:5])
def base_data_plots():
    # plots.all_plots(ds_loans[fico], "Fico Scores", "no unit")
    # plots.all_plots(ds_loans[int_rate], "Interest Rates", "%")
    # plots.all_plots(ds_loans[length], "Loan Length", "months")
    fig = plt.figure("Base data")
    ax1 = fig.add_subplot(221)
    ax1.hist(ds_loans[fico])
    ax1.set_title("FICO scores")
    ax2 = fig.add_subplot(222)
    ax2.hist(ds_loans[int_rate])
    ax2.set_title("Interest Rate")
    ax3 = fig.add_subplot(223)
    ax3.hist(ds_loans[length])
    ax3.set_title("Loan Length (months)")
    plt.show()

# base_data_plots()

def scatter_matrix_plot():
    # spm = pd.scatter_matrix(ds_loans, figsize=(10,10), diagonal='hist', alpha=0.05)
    spm_reduced = pd.scatter_matrix(ds_loans[[fico, int_rate, length, amt_req, income]], figsize=(10,10), diagonal='hist', alpha=0.05)
    plt.show()
    
# scatter_matrix_plot()

#y=interest rate, x1=FICO score, x2=Loan amount
#transpose to have data in a vertical vector form
y = np.matrix(ds_loans[int_rate]).transpose()
x1 = np.matrix(ds_loans[fico]).transpose()
x2 = np.matrix(ds_loans[amt_req]).transpose()
#create a single matrix from x1 and x2
x = np.column_stack([x1, x2])
#add a constant to x to have the full equation: y = a1*x1 + a2*x2 + b
X = sm.add_constant(x)

#create the linear model and fit
model = sm.OLS(y, X).fit() #OLS: Ordinary Least Square
print("\nInterest_Rate = Cst + a1*Fico + a2*Loan_Amount:\n")
print(model.summary())
# print(dir(model))

#get model parameters
(cst, a1, a2) = (model.params[0], model.params[1], model.params[2])
print("\nInterest_Rate = Cst + a1*Fico + a2*Loan_Amount")
print("\nModel parameters:\n\tCst = {0}\n\ta1 = {1}\n\ta2 = {2}".format(cst, a1, a2))

#create a modeled interest rate
ds_loans["Predicted_interest_rate"] = cst + a1*x1 + a2*x2
# int_rate_model = cst + a1*x1 + a2*x2
def scatter_data_vs_model(predicted, ylabel):
    plt.scatter(ds_loans[int_rate], predicted)
    # plt.plot(ds_loans[int_rate], ds_loans[int_rate], color='r')
    plt.plot([0, 0.3], [0, 0.3], color='r', linewidth=2)
    plt.xlim([0, 0.3])
    plt.ylim([0, 0.3])
    plt.xlabel("Interest rate from data")
    plt.ylabel(ylabel)

# scatter_data_vs_model(ds_loans["Predicted_interest_rate"], "Model1")
# plt.show()
    
#Linear regression with one more input: monthly income
x3 = np.matrix(ds_loans[income]).transpose()
x_2 = np.column_stack([x1, x2, x3])
X_2 = sm.add_constant(x_2)
model_2 = sm.OLS(y, X_2).fit()
print(model_2.summary())
(cst_2, a1_2, a2_2, a3_2) = (model_2.params[0], model_2.params[1], model_2.params[2], model_2.params[3])
print("\nInterest_Rate = Cst + a1*Fico + a2*Loan_Amount + a3*Montthly_Income")
print("\nModel parameters:\n\tCst = {0}\n\ta1 = {1}\n\ta2 = {2}\n\ta3 = {3}".format(cst_2, a1_2, a2_2, a3_2))
ds_loans["Predicted_interest_rate_2"] = cst_2 + a1_2*x1 + a2_2*x2 + a3_2*x3
# scatter_data_vs_model(ds_loans["Predicted_interest_rate_2"], "Model2")
# plt.show()
# print(model_2.)

fig = plt.figure()
plt.subplot(1,2,1)
scatter_data_vs_model(ds_loans["Predicted_interest_rate"], "Model1 - LR(FICO, Loan Amount)")
fig.text(.15, .83, "R^2 = {0}".format(round(model.rsquared, 4)))
plt.subplot(1,2,2)
scatter_data_vs_model(ds_loans["Predicted_interest_rate_2"], "Model2 - LR(FICO, Loan Amount, Monthly Income)")
fig.text(.58, .83, "R^2 = {0}".format(round(model_2.rsquared, 4)))
plt.show()