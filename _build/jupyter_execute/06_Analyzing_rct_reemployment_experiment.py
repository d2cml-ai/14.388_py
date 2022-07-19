#!/usr/bin/env python
# coding: utf-8

# * Python code replication of:
# " https://www.kaggle.com/victorchernozhukov/analyzing-rct-reemployment-experiment "
# * Created by: Alexander Quispe and Anzony Quispe 

# # Analyzing RCT data with Precision Adjustment

# ## Data
# 
# In this lab, we analyze the Pennsylvania re-employment bonus experiment, which was previously studied in "Sequential testing of duration data: the case of the Pennsylvania ‘reemployment bonus’ experiment" (Bilias, 2000), among others. These experiments were conducted in the 1980s by the U.S. Department of Labor to test the incentive effects of alternative compensation schemes for unemployment insurance (UI). In these experiments, UI claimants were randomly assigned either to a control group or one of five treatment groups. Actually, there are six treatment groups in the experiments. Here we focus on treatment group 4, but feel free to explore other treatment groups. In the control group the current rules of the UI applied. Individuals in the treatment groups were offered a cash bonus if they found a job within some pre-specified period of time (qualification period), provided that the job was retained for a specified duration. The treatments differed in the level of the bonus, the length of the qualification period, and whether the bonus was declining over time in the qualification period; see http://qed.econ.queensu.ca/jae/2000-v15.6/bilias/readme.b.txt for further details on data. 
#   

# In[1]:


import pandas as pd
import pyreadr


# In[2]:


## loading the data
Penn = pd.read_csv("../data/penn_jae.dat" , sep='\s', engine='python')
n = Penn.shape[0]
p_1 = Penn.shape[1]
Penn = Penn[ (Penn['tg'] == 2) | (Penn['tg'] == 0) ]


# In[5]:


Penn.shape


# In[4]:


# Dependent variable
Penn['T4'] = (Penn[['tg']]==4).astype(int)

# Create category variable
Penn['dep'] = Penn['dep'].astype( 'category' )
Penn.head()


# In[5]:


Penn['dep'].unique()


# ### Model 
# To evaluate the impact of the treatments on unemployment duration, we consider the linear regression model:
# 
# $$
# Y =  D \beta_1 + W'\beta_2 + \varepsilon, \quad E \varepsilon (D,W')' = 0,
# $$
# 
# where $Y$ is  the  log of duration of unemployment, $D$ is a treatment  indicators,  and $W$ is a set of controls including age group dummies, gender, race, number of dependents, quarter of the experiment, location within the state, existence of recall expectations, and type of occupation.   Here $\beta_1$ is the ATE, if the RCT assumptions hold rigorously.
# 
# 
# We also consider interactive regression model:
# 
# $$
# Y =  D \alpha_1 + D W' \alpha_2 + W'\beta_2 + \varepsilon, \quad E \varepsilon (D,W', DW')' = 0,
# $$
# where $W$'s are demeaned (apart from the intercept), so that $\alpha_1$ is the ATE, if the RCT assumptions hold rigorously.

# Under RCT, the projection coefficient $\beta_1$ has
# the interpretation of the causal effect of the treatment on
# the average outcome. We thus refer to $\beta_1$ as the average
# treatment effect (ATE). Note that the covariates, here are
# independent of the treatment $D$, so we can identify $\beta_1$ by
# just linear regression of $Y$ on $D$, without adding covariates.
# However we do add covariates in an effort to improve the
# precision of our estimates of the average treatment effect.

# ### Analysis
# 
# We consider 
# 
# *  classical 2-sample approach, no adjustment (CL)
# *  classical linear regression adjustment (CRA)
# *  interactive regression adjusment (IRA)
# 
# and carry out robust inference using the *estimatr* R packages. 

# # Carry out covariate balance check

# This is done using "lm_robust" command which unlike "lm" in the base command automatically does the correct Eicher-Huber-White standard errors, instead othe classical non-robus formula based on the homoscdedasticity command.

# In[6]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
import numpy as np


# ## Regress treatment on all covariates
# I use data from R

# In[7]:


y = Penn[['T4']].reset_index( drop = True )


# In[8]:


# Get data from R
result = pyreadr.read_r('../data/m_reg.RData')
X_vars = result['X1']


# In[9]:


# model = "T4~(female+black+othrace+C(dep)+q2+q3+q4+q5+q6+agelt35+agegt54+durable+lusd+husd)**2"
model_results = sm.OLS( y, X_vars ).fit().get_robustcov_results(cov_type = "HC1")

print(model_results.summary())
print( "Number of regressors in the basic model:",len(model_results.params), '\n')


# We see that that even though this is a randomized experiment, balance conditions are failed.

# # Model Specification
# I use data from R.

# In[10]:


# model specifications
# take log of inuidur1
Penn["log_inuidur1"] = np.log( Penn["inuidur1"] ) 
log_inuidur1 = pd.DataFrame(np.log( Penn["inuidur1"] ) ).reset_index( drop = True )

# no adjustment (2-sample approach)
formula_cl = 'log_inuidur1 ~ T4'

# adding controls
# formula_cra = 'log_inuidur1 ~ T4 + (female+black+othrace+dep+q2+q3+q4+q5+q6+agelt35+agegt54+durable+lusd+husd)**2'
# Omitted dummies: q1, nondurable, muld

ols_cl = smf.ols( formula = formula_cl, data = Penn ).fit().get_robustcov_results(cov_type = "HC1")

#getting data
# Get data from R
result = pyreadr.read_r('../data/ols_cra_reg.RData')
X_vars = result['X1']

ols_cra = sm.OLS( log_inuidur1, X_vars ).fit().get_robustcov_results(cov_type = "HC1")

# Results 
print(ols_cl.summary())
print(ols_cra.summary())


# The interactive specificaiton corresponds to the approach introduced in Lin (2013).

# In[11]:


# create Y variable 
log_inuidur1 = pd.DataFrame(np.log( Penn["inuidur1"] )).reset_index( drop = True )


# In[12]:


# Reset index to estimation
# Get data from R
result = pyreadr.read_r('../data/ols_ira_reg.RData')
X_vars = result['S1']

ols_ira = sm.OLS( log_inuidur1, X_vars ).fit().get_robustcov_results(cov_type = "HC1")

# Results 
print(ols_ira.summary())


# # Next we try out partialling out with lasso

# In[6]:


import hdmpy


# Next we try out partialling out with lasso

# In[7]:


# Get data from R
result = pyreadr.read_r('../data/rlasso_ira_reg.RData')
X_vars = result['S']


# In[15]:


result = hdmpy.rlassoEffects( X_vars, log_inuidur1, index = 0 )       

rlasso_ira = pd.DataFrame(np.array( (result.res['coefficients'][0] , result.res['se'][0] ,            result.res['t'][0] , result.res['pval'][0] ) ).reshape(1, 4) , columns = ['Coef.' ,                             "Std.Err." , "t" , 'P>|t|'] , index = ['T4'])
rlasso_ira


# ### Results

# In[16]:


table2 = np.zeros((2, 4))
table2[0,0] = ols_cl.summary2().tables[1]['Coef.']['T4']
table2[0,1] = ols_cra.summary2().tables[1]['Coef.']['T4TRUE']
table2[0,2] = ols_ira.summary2().tables[1]['Coef.']['T4']
table2[0,3] = rlasso_ira['Coef.']['T4']

table2[1,0] = ols_cl.summary2().tables[1]['Std.Err.']['T4']
table2[1,1] = ols_cra.summary2().tables[1]['Std.Err.']['T4TRUE']
table2[1,2] = ols_ira.summary2().tables[1]['Std.Err.']['T4']
table2[1,3] = rlasso_ira['Std.Err.']['T4']

table2 = pd.DataFrame(table2, columns = ["$CL$", "$CRA$", "$IRA$", "$IRA Lasso$"],                       index = ["estimate","standard error"])
table2


# Treatment group 4 experiences an average decrease of about $7.8\%$ in the length of unemployment spell.
# 
# 
# Observe that regression estimators delivers estimates that are slighly more efficient (lower standard errors) than the simple 2 mean estimator, but essentially all methods have very similar standard errors. From IRA results we also see that there is not any statistically detectable heterogeneity.  We also see the regression estimators offer slightly lower estimates -- these difference occur perhaps to due minor imbalance in the treatment allocation, which the regression estimators try to correct.
# 
# 
# 
