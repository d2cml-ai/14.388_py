#!/usr/bin/env python
# coding: utf-8

# * Python code replication of: " https://www.kaggle.com/janniskueck/pm1-notebook1-prediction-newdata "
# * Created by: Alexander Quispe

# This notebook contains an example for teaching.

# # Introduction

# In labor economics an important question is what determines the wage of workers. This is a causal question, but we could begin to investigate from a predictive perspective.
# 
# In the following wage example,  Y  is the hourly wage of a worker and  X  is a vector of worker's characteristics, e.g., education, experience, gender. Two main questions here are:
# 
# * How to use job-relevant characteristics, such as education and experience, to best predict wages?
# 
# * What is the difference in predicted wages between men and women with the same job-relevant characteristics?
# 
# In this lab, we focus on the prediction question first.

# # Data

# The data set we consider is from the March Supplement of the U.S. Current Population Survey, year 2015. We select white non-hispanic individuals, aged 25 to 64 years, and working more than 35 hours per week during at least 50 weeks of the year. We exclude self-employed workers; individuals living in group quarters; individuals in the military, agricultural or private household sectors; individuals with inconsistent reports on earnings and employment status; individuals with allocated or missing information in any of the variables used in the analysis; and individuals with hourly wage below  3 .
# 
# The variable of interest  Y  is the hourly wage rate constructed as the ratio of the annual earnings to the total number of hours worked, which is constructed in turn as the product of number of weeks worked and the usual number of hours worked per week. In our analysis, we also focus on single (never married) workers. The final sample is of size  **n=5150** .

# # Data analysis
# 

# We start by loading the data set.

# In[1]:


# Import relevant packages
import pandas as pd
import numpy as np
import pyreadr


# In[2]:


rdata_read = pyreadr.read_r("../data/wage2015_subsample_inference.Rdata")
data = rdata_read[ 'data' ]
type(data)
data.shape


# Let's have a look at the structure of the data.

# In[3]:


data.info()


# In[4]:


data.describe()


# We are constructing the output variable  **Y**  and the matrix  **Z**  which includes the characteristics of workers that are given in the data.

# In[5]:


Y = np.log2(data['wage']) 
n = len(Y)
z = data.loc[:, ~data.columns.isin(['wage', 'lwage','Unnamed: 0'])]
p = z.shape[1]

print("Number of observation:", n, '\n')
print( "Number of raw regressors:", p)


# For the outcome variable *wage* and a subset of the raw regressors, we calculate the empirical mean to get familiar with the data.

# In[6]:


Z_subset = data.loc[:, data.columns.isin(["lwage","sex","shs","hsg","scl","clg","ad","mw","so","we","ne","exp1"])]
table = Z_subset.mean(axis=0)
table


# In[7]:


table = pd.DataFrame(data=table, columns={"Sample mean":"0"} )
table.index
index1 = list(table.index)
index2 = ["Log Wage","Sex","Some High School","High School Graduate",          "Some College","College Graduate", "Advanced Degree","Midwest",          "South","West","Northeast","Experience"]


# In[8]:


table = table.rename(index=dict(zip(index1,index2)))
table


# E.g., the share of female workers in our sample is ~44% ($sex=1$ if female).

# Alternatively, we can also print the table as latex.

# ## Prediction Question

# Now, we will construct a prediction rule for hourly wage $Y$, which depends linearly on job-relevant characteristics $X$:
# 
# \begin{equation}\label{decompose}
# Y = \beta'X+ \epsilon.
# \end{equation}

# Our goals are
# 
# * Predict wages  using various characteristics of workers.
# 
# * Assess the predictive performance using the (adjusted) sample MSE, the (adjusted) sample $R^2$ and the out-of-sample MSE and $R^2$.
# 
# 
# We employ two different specifications for prediction:
# 
# 
# 1. Basic Model:   $X$ consists of a set of raw regressors (e.g. gender, experience, education indicators,  occupation and industry indicators, regional indicators).
# 
# 
# 2. Flexible Model:  $X$ consists of all raw regressors from the basic model plus occupation and industry indicators, transformations (e.g., ${exp}^2$ and ${exp}^3$) and additional two-way interactions of polynomial in experience with other regressors. An example of a regressor created through a two-way interaction is *experience* times the indicator of having a *college degree*.
# 
# Using the **Flexible Model**, enables us to approximate the real relationship by a
#  more complex regression model and therefore to reduce the bias. The **Flexible Model** increases the range of potential shapes of the estimated regression function. In general, flexible models often deliver good prediction accuracy but give models which are harder to interpret.

# Now, let us fit both models to our data by running ordinary least squares (ols):

# In[9]:


# Import packages for OLS regression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. basic model
basic = 'lwage ~ sex + exp1 + shs + hsg+ scl + clg + mw + so + we + occ2+ ind2'
basic_results = smf.ols(basic , data=data).fit()
print(basic_results.summary()) # estimated coefficients
print( "Number of regressors in the basic model:",len(basic_results.params), '\n')  # number of regressors in the Basic Model


# ##### Note that the basic model consists of $51$ regressors.

# In[10]:


# 2. flexible model
flex = 'lwage ~ sex + shs+hsg+scl+clg+occ2+ind2+mw+so+we + (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we)'
flex_results_0 = smf.ols(flex , data=data)
flex_results = smf.ols(flex , data=data).fit()
print(flex_results.summary()) # estimated coefficients
print( "Number of regressors in the basic model:",len(flex_results.params), '\n') # number of regressors in the Flexible Model


# Note that the flexible model consists of $246$ regressors.

# ## Try Lasso next

# In[11]:


# Import relevant packages for lasso 
from sklearn.linear_model import LassoCV
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


# In[12]:


# Get exogenous variables from flexible model
X = flex_results_0.exog
X.shape


# In[13]:


# Set endogenous variable
lwage = data["lwage"]
lwage.shape


# In[14]:


alpha=0.1


# In[15]:


# Set penalty value = 0.1
#reg = linear_model.Lasso(alpha=0.1/np.log(len(lwage)))
reg = linear_model.Lasso(alpha = alpha)

# LASSO regression for flexible model
reg.fit(X, lwage)
lwage_lasso_fitted = reg.fit(X, lwage).predict( X )

# coefficients 
reg.coef_
print('Lasso Regression: R^2 score', reg.score(X, lwage))


# In[16]:


# Check predicted values
lwage_lasso_fitted


# Now, we can evaluate the performance of both models based on the (adjusted) $R^2_{sample}$ and the (adjusted) $MSE_{sample}$:

# In[17]:


# Basic Model
basic = 'lwage ~ sex + exp1 + shs + hsg+ scl + clg + mw + so + we + occ2+ ind2'
basic_results = smf.ols(basic , data=data).fit()

# Flexible model 
flex = 'lwage ~ sex + shs+hsg+scl+clg+occ2+ind2+mw+so+we + (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we)'
flex_results = smf.ols(flex , data=data).fit()


# In[18]:


# Assess the predictive performance
R2_1 = basic_results.rsquared
print("R-squared for the basic model: ", R2_1, "\n")
R2_adj1 = basic_results.rsquared_adj
print("adjusted R-squared for the basic model: ", R2_adj1, "\n")


R2_2 = flex_results.rsquared
print("R-squared for the basic model: ", R2_2, "\n")
R2_adj2 = flex_results.rsquared_adj
print("adjusted R-squared for the basic model: ", R2_adj2, "\n")

R2_L = reg.score(flex_results_0.exog, lwage)
print("R-squared for LASSO: ", R2_L, "\n")
R2_adjL = 1 - (1-R2_L)*(len(lwage)-1)/(len(lwage)-X.shape[1]-1)
print("adjusted R-squared for LASSO: ", R2_adjL, "\n")


# In[19]:


# calculating the MSE
MSE1 =  np.mean(basic_results.resid**2)
print("MSE for the basic model: ", MSE1, "\n")
p1 = len(basic_results.params) # number of regressors
n = len(lwage)
MSE_adj1  = (n/(n-p1))*MSE1
print("adjusted MSE for the basic model: ", MSE_adj1, "\n")

MSE2 =  np.mean(flex_results.resid**2)
print("MSE for the flexible model: ", MSE2, "\n")
p2 = len(flex_results.params) # number of regressors
n = len(lwage)
MSE_adj2  = (n/(n-p2))*MSE2
print("adjusted MSE for the flexible model: ", MSE_adj2, "\n")


MSEL = mean_squared_error(lwage, lwage_lasso_fitted)
print("MSE for the LASSO model: ", MSEL, "\n")
pL = reg.coef_.shape[0] # number of regressors
n = len(lwage)
MSE_adjL  = (n/(n-pL))*MSEL
print("adjusted MSE for LASSO model: ", MSE_adjL, "\n")


# In[20]:


# Package for latex table 
import array_to_latex as a2l

table = np.zeros((3, 5))
table[0,0:5] = [p1, R2_1, MSE1, R2_adj1, MSE_adj1]
table[1,0:5] = [p2, R2_2, MSE2, R2_adj2, MSE_adj2]
table[2,0:5] = [pL, R2_L, MSEL, R2_adjL, MSE_adjL]


# In[28]:


table = pd.DataFrame(table, columns = ["p","$R^2_{sample}$","$MSE_{sample}$","$R^2_{adjusted}$", "$MSE_{adjusted}$"],                       index = ["basic reg","flexible reg", "lasso flex"])
table


# Considering all measures above, the flexible model performs slightly better than the basic model. 
# 
# One procedure to circumvent this issue is to use **data splitting** that is described and applied in the following.

# ## Data Splitting
# 
# Measure the prediction quality of the two models via data splitting:
# 
# - Randomly split the data into one training sample and one testing sample. Here we just use a simple method (stratified splitting is a more sophisticated version of splitting that we can consider).
# - Use the training sample for estimating the parameters of the Basic Model and the Flexible Model.
# - Use the testing sample for evaluation. Predict the $\mathtt{wage}$  of every observation in the testing sample based on the estimated parameters in the training sample.
# - Calculate the Mean Squared Prediction Error $MSE_{test}$ based on the testing sample for both prediction models. 

# In[29]:


# Import relevant packages for splitting data
import random
import math

# Set Seed
# to make the results replicable (generating random numbers)
np.random.seed(0)
random = np.random.randint(0,n, size=math.floor(n))
data["random"] = random
random    # the array does not change 


# In[30]:


data_2 = data.sort_values(by=['random'])
data_2.head()


# In[31]:


# Create training and testing sample 
train = data_2[ : math.floor(n*4/5)]    # training sample
test =  data_2[ math.floor(n*4/5) : ]   # testing sample
print(train.shape)
print(test.shape)


# In[32]:


# Basic Model
basic = 'lwage ~ sex + exp1 + shs + hsg+ scl + clg + mw + so + we + occ2+ ind2'
basic_results = smf.ols(basic , data=data).fit()

# Flexible model 
flex = 'lwage ~ sex + shs+hsg+scl+clg+occ2+ind2+mw+so+we + (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we)'
flex_results = smf.ols(flex , data=data).fit()


# In[33]:


# basic model
# estimating the parameters in the training sample
basic_results = smf.ols(basic , data=train).fit()
print(basic_results.summary())


# In[34]:


lwage_test = test["lwage"].values
#test = test.drop(columns=['wage', 'lwage', 'random'])
#test


# In[35]:


# calculating the out-of-sample MSE
test = sm.add_constant(test)   #add constant 

lwage_pred =  basic_results.predict(test) # predict out of sample
print(lwage_pred)


# In[36]:


MSE_test1 = np.sum((lwage_test-lwage_pred)**2)/len(lwage_test)
R2_test1  = 1 - MSE_test1/np.var(lwage_test)

print("Test MSE for the basic model: ", MSE_test1, " ")
print("Test R2 for the basic model: ", R2_test1)


# In the basic model, the $MSE_{test}$ is quite closed to the $MSE_{sample}$.

# In[37]:


# Flexible model
# estimating the parameters in the training sample
flex_results = smf.ols(flex , data=train).fit()

# calculating the out-of-sample MSE
lwage_flex_pred =  flex_results.predict(test) # predict out of sample
lwage_test = test["lwage"].values

MSE_test2 = np.sum((lwage_test-lwage_flex_pred)**2)/len(lwage_test)
R2_test2  = 1 - MSE_test2/np.var(lwage_test)

print("Test MSE for the flexible model: ", MSE_test2, " ")
print("Test R2 for the flexible model: ", R2_test2)


# In the flexible model, the discrepancy between the $MSE_{test}$ and the $MSE_{sample}$ is not large.

# It is worth to notice that the $MSE_{test}$ vary across different data splits. Hence, it is a good idea average the out-of-sample MSE over different data splits to get valid results.
# 
# Nevertheless, we observe that, based on the out-of-sample $MSE$, the basic model using ols regression performs is about as well (or slightly better) than the flexible model. 
# 
# 
# Next, let us use lasso regression in the flexible model instead of ols regression. Lasso (*least absolute shrinkage and selection operator*) is a penalized regression method that can be used to reduce the complexity of a regression model when the number of regressors $p$ is relatively large in relation to $n$. 
# 
# Note that the out-of-sample $MSE$ on the test sample can be computed for any other black-box prediction method as well. Thus, let us finally compare the performance of lasso regression in the flexible model to ols regression.

# In[38]:


# flexible model using lasso
# get exogenous variables from training data used in flex model
flex_results_0 = smf.ols(flex , data=train)
X_train = flex_results_0.exog
print(X_train.shape)

# Get endogenous variable 
lwage_train = train["lwage"]
print(lwage_train.shape)


# In[39]:


# flexible model using lasso
# get exogenous variables from testing data used in flex model
flex_results_1 = smf.ols(flex , data=test)
X_test = flex_results_1.exog
print(X_test.shape)

# Get endogenous variable 
lwage_test = test["lwage"]
print(lwage_test.shape)


# In[40]:


# calculating the out-of-sample MSE
reg = linear_model.Lasso(alpha=0.1)
lwage_lasso_fitted = reg.fit(X_train, lwage_train).predict( X_test )

MSE_lasso = np.sum((lwage_test-lwage_lasso_fitted)**2)/len(lwage_test)
R2_lasso  = 1 - MSE_lasso/np.var(lwage_test)

print("Test MSE for the flexible model: ", MSE_lasso, " ")
print("Test R2 for the flexible model: ", R2_lasso)


# Finally, let us summarize the results:

# In[42]:


# Package for latex table 
import array_to_latex as a2l

table2 = np.zeros((3, 2))
table2[0,0] = MSE_test1
table2[1,0] = MSE_test2
table2[2,0] = MSE_lasso
table2[0,1] = R2_test1
table2[1,1] = R2_test2
table2[2,1] = R2_lasso

table2 = pd.DataFrame(table2, columns = ["$MSE_{test}$", "$R^2_{test}$"],                       index = ["basic reg","flexible reg","lasso regression"])
table2

