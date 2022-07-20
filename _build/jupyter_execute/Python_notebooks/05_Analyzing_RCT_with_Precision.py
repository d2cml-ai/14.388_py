#!/usr/bin/env python
# coding: utf-8

# * Python code replication of:
# " https://www.kaggle.com/victorchernozhukov/r-notebook-analyzing-rct-with-precision "
# * Created by: Alexander Quispe and Anzony Quispe 

# # Analyzing RCT with Precision by Adjusting for Baseline Covariates

# # Jonathan Roth's DGP
# 
# Here we set up a DGP with heterogenous effects. In this example, with is due to Jonathan Roth, we have
# $$
# E [Y(0) | Z] = - Z, \quad E [Y(1) |Z] = Z, \quad Z \sim N(0,1).
# $$
# The CATE is
# $$
# E [Y(1) - Y(0) | Z ]= 2 Z.
# $$
# and the ATE is
# $$
# 2 E Z = 0.
# $$
# 
# We would like to estimate ATE as precisely as possible.
# 
# An economic motivation for this example could be provided as follows: Let D be the treatment of going to college, and $Z$ academic skills.  Suppose that academic skills cause lower earnings Y(0) in jobs that don't require college degree, and cause higher earnings  Y(1) in jobs that require college degrees. This type of scenario is reflected in the DGP set-up above.
# 
# 

# In[1]:


# Import relevant packages for splitting data
import numpy as np
import random
import math
import pandas as pd

# Set Seed
# to make the results replicable (generating random numbers)
np.random.seed(12345676)     # set MC seed

n = 1000                # sample size
Z = np.random.normal(0, 1, 1000).reshape((1000, 1))  # generate Z
Y0 = -Z + np.random.normal(0, 1, 1000).reshape((1000, 1))   # conditional average baseline response is -Z
Y1 = Z + np.random.normal(0, 1, 1000).reshape((1000, 1))    # conditional average treatment effect is +Z
D = (np.random.uniform(0, 1, n)<.2).reshape((1000, 1))      # treatment indicator; only 20% get treated
np.mean(D)


# In[2]:


Y = Y1*D + Y0*(1-D)  # observed Y
D = D - np.mean(D)      # demean D
Z = Z - np.mean(Z)        # demean Z


# # Analyze the RCT data with Precision Adjustment
# 
# Consider 
# 
# *  classical 2-sample approach, no adjustment (CL)
# *  classical linear regression adjustment (CRA)
# *  interactive regression adjusment (IRA)
# 
# Carry out inference using robust inference, using the sandwich formulas (Eicker-Huber-White).  
# 
# Observe that CRA delivers estimates that are less efficient than CL (pointed out by Freedman), whereas IRA delivers more efficient approach (pointed out by Lin). In order for CRA to be more efficient than CL, we need the CRA to be a correct model of the conditional expectation function of Y given D and X, which is not the case here.

# In[3]:


Z_times_D = Z*D
X = np.hstack((D, Z, Z_times_D))
data = pd.DataFrame(X, columns = ["D", "Z", "Z_times_D"])
data


# In[4]:


# Import packages for OLS regression
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[5]:


CL_model = "Y ~ D"          
CRA_model = "Y ~ D + Z"      #classical
IRA_model = "Y ~ D+ Z+ Z*D" #interactive approach

CL = smf.ols(CL_model , data=data).fit()
CRA = smf.ols(CRA_model , data=data).fit()
IRA = smf.ols(IRA_model , data=data).fit()


# In[6]:


# Check t values of regressors 
print(CL.tvalues)
print(CRA.tvalues)
print(IRA.tvalues)


# # Using classical standard errors (non-robust) is misleading here.
# 
# We don't teach non-robust standard errors in econometrics courses, but the default statistical inference for lm() procedure in R, summary.lm(), still uses 100-year old concepts, perhaps in part due to historical legacy.  
# 
# Here the non-robust standard errors suggest that there is not much difference between the different approaches, contrary to the conclusions reached using the robust standard errors.

# In[7]:


# we are interested in the coefficients on variable "D".
print(CL.summary())
print(CRA.summary())
print(IRA.summary())


# In[122]:





# # Verify Asymptotic Approximations Hold in Finite-Sample Simulation Experiment

# In[8]:


np.random.seed(12345676)     # set MC seed
n = 1000
B = 1000

# numpy format of data = float32
CLs = np.repeat(0., B)
CRAs = np.repeat(0., B)
IRAs = np.repeat(0., B)

# models
CL_model = "Y ~ D"          
CRA_model = "Y ~ D + Z"      #classical
IRA_model = "Y ~ D+ Z+ Z*D" #interactive approachIRAs = np.repeat(0, B)

# simulation
for i in range(0, B, 1):
    Z = np.random.normal(0, 1, n).reshape((n, 1))
    Y0 = -Z + np.random.normal(0, 1, n).reshape((n, 1))
    Y1 = Z + np.random.normal(0, 1, n).reshape((n, 1))
    D = (np.random.uniform(0, 1, n)<.2).reshape((n, 1))
    
    D = D - np.mean(D)
    Z = Z - np.mean(Z)
    
    Y = Y1*D + Y0*(1-D)
    
    Z_times_D = Z*D
    X = np.hstack((D, Z, Z_times_D))
    data = pd.DataFrame(X, columns = ["D", "Z", "Z_times_D"])
     
    CLs[i,] = smf.ols(CL_model , data=data).fit().params[1]
    CRAs[i,] = smf.ols(CRA_model , data=data).fit().params[1]
    IRAs[i,] = smf.ols(IRA_model , data=data).fit().params[1]

# check  standard deviations
print("Standard deviations for estimators")
print(np.sqrt(np.mean(CLs**2)))
print(np.sqrt(np.mean(CRAs**2)))
print(np.sqrt(np.mean(IRAs**2)))

