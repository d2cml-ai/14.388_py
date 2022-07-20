#!/usr/bin/env python
# coding: utf-8

# This is a simple demonstration of Debiased Machine Learning estimator for the Conditional Average Treatment Effect. 
# Goal is to estimate the effect of 401(k) eligibility on net financial assets for each value of income. 
# Data set is the same as in (Chernozhukov, Hansen, 2004). 
# 
# 
# The method is based on the following paper. 
# 
# Title:  Debiased Machine Learning of Conditional Average Treatment Effect and Other Causal Functions
# 
# Authors: Semenova, Vira and Chernozhukov, Victor. 
# 
# Arxiv version: https://arxiv.org/pdf/1702.06240.pdf
# 
# Published version with replication code: https://academic.oup.com/ectj/advance-article/doi/10.1093/ectj/utaa027/5899048
# 
# 
# [1]Victor Chernozhukov and Christian Hansen. The impact of 401(k) participation on the wealth distribution: An instrumental quantile regression analysis. Review of Economics and Statistics, 86(3):735–751, 2004.

# Background
# 
# The target function is Conditional Average Treatment Effect, defined as 
# 
# $$ g(x)=E [ Y(1) - Y(0) |X=x], $$ 
# 
# where $Y(1)$ and $Y(0)$ are potential outcomes in treated and control group. In our case, $Y(1)$ is the potential Net Financial Assets if a subject is eligible for 401(k), and $Y(0)$ is the potential Net Financial Assets if a subject is ineligible. $X$ is a covariate of interest, in this case, income.
# $ g(x)$ shows expected effect of eligibility on NET TFA for a subject whose income level is $x$.
# 
# 
# 
# If eligibility indicator is independent of $Y(1), Y(0)$, given pre-401-k assignment characteristics $Z$, the function can expressed in terms of observed data (as opposed to hypothetical, or potential outcomes). Observed data consists of  realized NET TFA $Y = D Y(1) + (1-D) Y(0)$, eligibility indicator $D$, and covariates $Z$ which includes $X$, income. The expression for $g(x)$ is
# 
# $$ g(x) = E [ Y (\eta_0) \mid X=x], $$
# where the transformed outcome variable is
# 
# $$Y (\eta) = \dfrac{D}{s(Z)} \left( Y - \mu(1,Z) \right) - \dfrac{1-D}{1-s(Z)} \left( Y - \mu(0,Z) \right) + \mu(1,Z) - \mu(0,Z),$$
# 
# the probability of eligibility is 
# 
# $$s_0(z) = Pr (D=1 \mid Z=z),$$ 
# 
# the expected net financial asset given $D =d \in \{1,0\}$ and $Z=z$ is
# 
# $$ \mu(d,z) = E[ Y \mid Z=z, D=d]. $$
# 
# Our goal is to estimate $g(x)$.
# 
# 
# In step 1, we estimate the unknown functions $s_0(z),  \mu(1,z),  \mu(0,z)$ and plug them into $Y (\eta)$.
# 
# 
# In step 2, we approximate the function $g(x)$ by a linear combination of basis functions:
# 
# $$ g(x) = p(x)' \beta_0, $$
# 
# 
# where $p(x)$ is a vector of polynomials or splines and
# 
# $$ \beta_0 = (E p(X) p(X))^{-1} E p(X) Y (\eta_0) $$
# 
# is the best linear predictor. We report
# 
# $$
# \widehat{g}(x) = p(x)' \widehat{\beta},
# $$
# 
# where $\widehat{\beta}$ is the ordinary least squares estimate of $\beta_0$ defined on the random sample $(X_i, D_i, Y_i)_{i=1}^N$
# 
# $$
# 	\widehat{\beta} :=\left( \dfrac{1}{N} \sum_{i=1}^N p(X_i) p(X_i)' \right)^{-1} \dfrac{1}{N} \sum_{i=1}^N  p(X_i)Y_i(\widehat{\eta})
# $$

# In[1]:


import numpy as np
import pandas as pd
import doubleml as dml
from doubleml.datasets import fetch_401K

# Import relevant packages
import pyreadr
from sklearn import preprocessing
import patsy

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from xgboost import XGBClassifier, XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
import matplotlib.patches as mpatches


# In[3]:


# pension = fetch_401K(return_type='DataFrame')


# In[4]:


pension_Read = pyreadr.read_r("../data/pension.Rdata")
pension = pension_Read[ 'pension' ]


# In[5]:


pension["net_tfa"] = pension["net_tfa"] / 10000
pension


# In[6]:


## covariate of interest -- log income --
pension["inc"] = np.log(pension["inc"])

pension = pension[~pension.isin([np.nan, np.inf, -np.inf]).any(1)]
pension = pension.reset_index()


# In[7]:


## outcome variable -- total net financial assets
Y = pension["net_tfa"]

## binary treatment --  indicator of 401(k) eligibility
D = pension["e401"]

X = pension["inc"]

## raw covariates so that Y(1) and Y(0) are independent of D given Z
Z = pension[["age","inc","fsize","educ","male","db","marr","twoearn","pira","hown","hval","hequity","hmort",
              "nohs","hs","smcol"]]

Z = Z.to_numpy()

y_name = "net_tfa"
d_name = "e401"
form_z = "(poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + as.factor(marr) + as.factor(twoearn) + as.factor(db) + as.factor(pira) + as.factor(hown))^2"


# In[8]:


Y = Y.to_numpy()
D = D.to_numpy()
X = X.to_numpy()


# In[9]:


print("\n sample size is {} \n".format(len(Y)))
print("\n num raw covariates z is {} \n".format(Z.shape[1]))


# In[10]:


features = pension.copy()[['marr', 'twoearn', 'db', 'pira', 'hown']]


# In[11]:


poly_dict = {'age': 6,
             'inc': 8,
             'educ': 4,
             'fsize': 2}
for key, degree in poly_dict.items():
    poly = PolynomialFeatures(degree, include_bias=False)
    data_transf = poly.fit_transform(pension[[key]])
    x_cols = poly.get_feature_names([key])
    data_transf = pd.DataFrame(data_transf, columns=x_cols)
    
    features = pd.concat((features, data_transf),
                          axis=1, sort=False)


# In[12]:


import patsy 
from patsy import ModelDesc, Term, EvalFactor


# In[13]:


features


# In[14]:


new_columns = ['marr', 'twoearn', 'db', 'pira', 'hown',        'age', 'age_2', 'age_3', 'age_4', 'age_5', 'age_6',         'inc', 'inc_2', 'inc_3', 'inc_4', 'inc_5', 'inc_6', 'inc_7', 'inc_8',         'educ', 'educ_2', 'educ_3', 'educ_4',        'fsize', 'fsize_2']


# In[15]:


features.columns = [new_columns]
features


# In[18]:


formula = "(marr + twoearn + db + pira + hown + age + age_2 +  age_3 +  age_4 +  age_5 +  age_6 + inc +  inc_2 +  inc_3 + inc_4 +  inc_5 +  inc_6 +  inc_7 +  inc_8 + educ +  educ_2 + educ_3 + educ_4 + fsize +  fsize_2)**2"
formula


# In[19]:


y_name = pension["net_tfa"].to_numpy()
d_name = pension["e401"].to_numpy()
form_z = patsy.dmatrix(formula, features)


# In Step 1, we estimate three functions:
# 
# 1. probability of treatment assignment $s_0(z)$ 
# 
# 2.-3. regression functions $\mu_0(1,z)$ and $\mu_0(0,z)$.  
# 
# We use the cross-fitting procedure with $K=2$ holds. For definition of cross-fitting with $K$ folds, check the sample splitting in ```DML2.for.PLM``` function defined in https://www.kaggle.com/victorchernozhukov/debiased-ml-for-partially-linear-model-in-r
# 
# For each function, we try random forest.
# 

# First Stage: estimate $\mu_0(1,z)$ and $\mu_0(0,z)$ and $s_0(z)$ by lasso

# In[20]:


def first_stage_lasso(data,d_name,y_name, form_z, seed=1):

    # Sample size 
    N = form_z.shape[0]

    # Estimated regression function in control group
    mu0_hat = np.ones(N)

    # Estimated regression function in treated group
    mu1_hat = np.ones(N)

    # Propensity score
    s_hat = np.ones(N)

    seed = 1 
    np.random.seed(seed)

    ## define sample splitting
    inds_train_0 = np.random.choice(np.arange(0,N) , int(np.floor(N/2)), replace=False)
    #inds_eval_0 = np.setdiff1d(np.arange(0,N), inds_train)

    # Split data - index to keep are in mask as booleans
    include_idx = set(inds_train_0)  
    mask = np.array([(i in include_idx) for i in range(N)])

    inds_train = mask
    inds_eval = ~mask

    print("Estimate treatment probability, first half")

    ## conditional probability of 401 k eligibility (i.e., propensity score) based on logistic regression
    fitted_lasso_pscore = LogisticRegressionCV(cv=5, random_state=0).fit(form_z[inds_train,], d_name[inds_train,])
    s_hat[inds_eval,] = fitted_lasso_pscore.predict(form_z[inds_eval,])

    print ("Estimate treatment probability, second half")

    fitted_lasso_pscore = LogisticRegressionCV(cv=5, random_state=0).fit(form_z[inds_eval,], d_name[inds_eval,])
    s_hat[inds_train,] = fitted_lasso_pscore.predict(form_z[inds_train,])

    data1 = data
    data1["d_name"] = 1 

    data0 = data
    data0["d_name"] = 0

    # Create matrix with main covatiares
    d_name_form_z = np.concatenate( ( d_name.reshape(N,1) , form_z ), axis  =  1 )
    d_name_form_z_0 = np.concatenate( ( data0["d_name"].to_numpy().reshape(N,1) , form_z ) , axis  =  1 )
    d_name_form_z_1 = np.concatenate( ( data1["d_name"].to_numpy().reshape(N,1) , form_z ) , axis  =  1 )
    
    print("Estimate expectation function, first half") 

    fitted_lasso_mu = LassoCV(cv = 5 ,  max_iter=10000 ).fit( d_name_form_z[inds_train,], y_name[inds_train,] )
    mu1_hat[inds_eval,] = fitted_lasso_mu.predict(d_name_form_z_1[inds_eval,])
    mu0_hat[inds_eval,] = fitted_lasso_mu.predict(d_name_form_z_0[inds_eval,])

    print("Estimate expectation function, first half") 
    
    fitted_lasso_mu = LassoCV(cv = 5 ,  max_iter=10000 ).fit( d_name_form_z[inds_eval,], y_name[inds_eval,] )
    mu1_hat[inds_train,] = fitted_lasso_mu.predict(d_name_form_z_1[inds_train,])
    mu0_hat[inds_train,] = fitted_lasso_mu.predict(d_name_form_z_0[inds_train,])


    return mu1_hat, mu0_hat, s_hat


# First Stage: estimate $\mu_0(1,z)$ and $\mu_0(0,z)$ and $s_0(z)$ by random forest

# In[21]:


def first_stage_rf(Y,D,X,Z,seed=1):

    # Sample size 
    N = D.shape[0]

    # Estimated regression function in control group
    mu0_hat = np.ones(N)

    # Estimated regression function in treated group
    mu1_hat = np.ones(N)

    # Propensity score
    s_hat = np.ones(N)

    seed = 1 
    np.random.seed(seed)

    ## define sample splitting
    inds_train_0 = np.random.choice(np.arange(0,N) , int(np.floor(N/2)), replace=False)
    #inds_eval_0 = np.setdiff1d(np.arange(0,N), inds_train)

    # Split data - index to keep are in mask as booleans
    include_idx = set(inds_train_0)  
    mask = np.array([(i in include_idx) for i in range(N)])

    inds_train = mask
    inds_eval = ~mask

    print("Estimate treatment probability, first half")
    ## conditional probability of 401 k eligibility (i.e., propensity score) based on logistic regression
    # In case we want similar variable as R code 
    #D.astype("category")
    
    fitted_rf_pscore = RandomForestRegressor( random_state = 0 ).fit( Z[inds_train,], D[inds_train,] )
    s_hat[inds_eval,] = fitted_rf_pscore.predict(Z[inds_eval,])
    

    print ("Estimate treatment probability, second half")

    fitted_rf_pscore = RandomForestRegressor( random_state = 0 ).fit( Z[inds_eval,],D[inds_eval,] )
    s_hat[inds_train,] = fitted_rf_pscore.predict(Z[inds_train,])
    
    
    ## conditional expected net financial assets (i.e.,  regression function) based on random forest
    
#     data1 = data
#     data1["d_name"] = 1 

#     data0 = data
#     data0["d_name"] = 0

    # Create matrix with main covatiares
    covariates = np.concatenate( ( Z , D.reshape(N,1) ), axis  =  1 )
    covariates1 = np.concatenate( ( Z, np.ones(N).reshape(N,1) ) , axis  =  1 )
    covariates0 = np.concatenate( ( Z, np.zeros(N).reshape(N,1) ) , axis  =  1 )
    
    print("Estimate expectation function, first half") 

    fitted_rf_mu = RandomForestRegressor( random_state = 0 ).fit( covariates[inds_train,], Y[inds_train,] )
    mu1_hat[inds_eval,] = fitted_rf_mu.predict(covariates1[inds_eval,])
    mu0_hat[inds_eval,] = fitted_rf_mu.predict(covariates0[inds_eval,])
    

    print("Estimate expectation function, second half") 
    
    fitted_rf_mu = RandomForestRegressor( random_state = 0 ).fit( covariates[inds_eval,], Y[inds_eval,] )
    mu1_hat[inds_train,] = fitted_rf_mu.predict(covariates1[inds_train,])
    mu0_hat[inds_train,] = fitted_rf_mu.predict(covariates0[inds_train,])


    return mu1_hat, mu0_hat, s_hat


# In Step 2, we approximate $Y(\eta_0)$ by a vector of basis functions. There are two use cases:
# ****
# 2.A. Group Average Treatment Effect, described above
# 
# 
# 2.B. Average Treatment Effect conditional on income value. There are three smoothing options:
# 
# 1. splines offered in ```least_squares_splines```
# 
# 2. orthogonal polynomials with the highest degree chosen by cross-validation ```least_squares_series```
# 
# 3. standard polynomials with the highest degree input by user ```least_squares_series_old```
# 
# 
# The default option is option 3.

# 2.A. The simplest use case of Conditional Average Treatment Effect is GATE, or Group Average Treatment Effect. Partition the support of income as
# 
# $$ - \infty = \ell_0 < \ell_1 < \ell_2 \dots \ell_K = \infty $$
# 
# define intervals $I_k = [ \ell_{k-1}, \ell_{k})$. Let $X$ be income covariate. For $X$, define a group indicator 
# 
# $$ G_k(X) = 1[X \in I_k], $$
# 
# and the vector of basis functions 
# 
# $$ p(X) = (G_1(X), G_2(X), \dots, G_K(X)) $$
# 
# Then, the Best Linear Predictor $\beta_0$ vector shows the average treatment effect for each group.

# In[22]:


## estimate first stage functions by random forest
## may take a while
fs_hat_rf = first_stage_rf(Y,D,X,Z,seed = 1)


# In[25]:


mu1_hat, mu0_hat, s_hat = fs_hat_rf


# In[26]:


RobustSignal = (Y - mu1_hat)*D/s_hat - (Y - mu0_hat)*(1-D)/(1-s_hat) + mu1_hat - mu0_hat
RobustSignal


# In[27]:


def qtmax(C, S, alpha):
    
    p = C.shape[0]
    random_matrix = np.reshape(np.random.normal(0,1,p*S), (p, S))
    tmaxs = np.max(np.abs(random_matrix), axis = 0)
    quantile = np.quantile(tmaxs,1 - alpha)
    
    return quantile


# In[28]:


def group_average_treatment_effect(X, Y, max_grid = 5, alpha = 0.05, B = 10000):
    probs = np.arange(0, max_grid+1)/max_grid
    grid = np.quantile(X, probs)
    
    X_raw = np.empty((len(Y), len(grid)-1,))
    
    for k in range(1, len(grid)):
        test = lambda x : 1 if (x>=grid[k-1] and x<grid[k]) else 0
        X_raw[:, k-1] = np.array([test(xi) for xi in X])

    k = len(grid) - 1
    test = lambda x : 1 if (x>=grid[k-1] and x<=grid[k]) else 0
    X_raw[:, k-1] = np.array([test(xi) for xi in X])
    
    # OLS regression
    ols_fit  = sm.OLS(Y, X_raw).fit(cov_type='HC2')
    coefs = ols_fit.summary2().tables[1]['Coef.']
    vars = ols_fit.summary2().tables[1].index.values.tolist()
    HCV_coefs = ols_fit.cov_params()
    coefs_se = np.sqrt(np.diag(HCV_coefs))
    
    ## this is an identity matrix
                     ## qtmax is simplified
    C_coefs = np.diag(1/np.sqrt(np.diag(HCV_coefs)))*HCV_coefs*np.diag(1/np.sqrt(np.diag(HCV_coefs)))
    tes = coefs
    tes_se = coefs_se
    tes_cor = C_coefs
    crit_val = qtmax(tes_cor,B,alpha)

    tes_ucb = tes + crit_val * tes_se
    tes_lcb = tes - crit_val * tes_se

    tes_uci = tes + norm.ppf(1- alpha/2)*tes_se
    tes_lci = tes + norm.ppf(alpha/2)*tes_se

    return coefs, tes_lci, tes_uci, tes_lcb, tes_ucb, crit_val    


# In[29]:


res = group_average_treatment_effect(X=X,Y=RobustSignal,max_grid = 5)
res[1]


# In[30]:


## this code is taken from L1 14.382 taught at MIT
## author: Mert Demirer
#options(repr.plot.width=10, repr.plot.height=8)

tes = res[0]
tes_lci = res[1]
tes_uci = res[2]

tes_lcb = res[3]
tes_ucb = res[4]
tes_lev = ['0%-20%', '20%-40%','40%-60%','60%-80%','80%-100%']


##### FINISH THE PLOT  PLEASE 
# specify the location of (left,bottom),width,height
rect=mpatches.Rectangle((1-0.2,1+0.2),tes_lci[0],7, 
                        fill = False,
                        color = "purple",
                        linewidth = 2)


# In[166]:


## this code is taken from L1 14.382 taught at MIT
## author: Mert Demirer
#options(repr.plot.width=10, repr.plot.height=8)

tes = np.array([0.438, 0.33255, 0.5623, 0.8037, 1.430])
tes_lci = np.array([0.38233, 0.22077, 0.444737, 0.61626, 1.0444])
tes_uci = np.array([0.4944, 0.44433, 0.679887, 0.991225, 1.81643])

tes_lcb = np.array([0.36529, 0.18679, 0.40900, 0.559282, 0.927107])
tes_ucb = np.array([0.5115, 0.478307, 0.7156, 1.0482107, 1.9337566])
tes_lev = ('0%-20%', '20%-40%','40%-60%','60%-80%','80%-100%')


##### FINISH THE PLOT  PLEASE 


# In[33]:


import matplotlib.pyplot as plt


# In[51]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches


# In[178]:


# Create figure and axes
fig = plt.figure(figsize=(10, 8), dpi=80)
ax = plt.subplot(111)
ax.set_xlim( 0.6, 5.4)
ax.set_ylim( 0.05, 2.09)
# Create a Rectangle patch

for i in range(1, 6):
    rect1 = patches.Rectangle((i-0.2, tes_lci[i-1]), 0.4, tes_uci[i-1] - tes_lci[i-1] , linewidth=3, color = None,                          fill = None, edgecolor='red')
    ax.add_patch(rect1)

    rect2 = patches.Rectangle((i-0.2, tes_lcb[i-1]), 0.4, tes_ucb[i-1] - tes_lcb[i-1] , linewidth=3, color = None,                          fill = None, edgecolor='blue')

    # Add the patch to the Axes
    ax.add_patch(rect2)
    
    ax.plot([i-0.2, i+0.2], [tes[i-1], tes[i-1]], 'k-', lw=2)

ax.legend(['Regression Estimate', '95% Simultaneous Confidence Interval', '95% Pointwise Confidence Interval'],           bbox_to_anchor=(0.3, 0.8), loc=3 )

leg = ax.get_legend()

leg.legendHandles[0].set_color('black')
leg.legendHandles[1].set_color('blue')
leg.legendHandles[2].set_color('red')

ax.set_ylabel("Average Effect on NET TFA (per 10 K)")
ax.set_xlabel("Income group")
ax.set_title("Group Average Treatment Effects on NET TFA")
ax.set_xticks(np.arange(1,6) )
ax.set_xticklabels(tes_lev)
plt.tight_layout()
plt.show()


# In[142]:





# In[117]:


axbox.x0


# In[37]:


dir(ax)


# In[ ]:


def least_squares_splines(X,Y,max_knot,norder,nderiv):
    ## Create technical regressors
    cs_bsp = np.zeros(max_knot - 1)
    for knot in range(1:max_knot+1 )
    
    
    


# In[724]:


max_knot = 5 
cs_bsp = np.zeros(max_knot - 1)
for knot in range(2, max_knot+1 ):
    probs = np.arange(0, max_knot+1)/max_knot
    breaks = np.quantile(X, probs)


# In[ ]:





# In[ ]:





# In[ ]:





# In[702]:


# max_grid = 5
# probs = np.arange(0, max_grid+1)/max_grid
# grid = np.quantile(X, probs)

# X_raw = np.empty((len(Y), len(grid)-1,))

# for k in range(1, len(grid)):
#     test = lambda x : 1 if (x>=grid[k-1] and x<grid[k]) else 0
#     X_raw[:, k-1] = np.array([test(xi) for xi in X])

# k = len(grid) - 1
# test = lambda x : 1 if (x>=grid[k-1] and x<=grid[k]) else 0
# X_raw[:, k-1] = np.array([test(xi) for xi in X])

# # OLS regression
# ols_fit  = sm.OLS(Y, X_raw).fit(cov_type='HC2')
# coefs = ols_fit.summary2().tables[1]['Coef.']
# vars = ols_fit.summary2().tables[1].index.values.tolist()
# HCV_coefs = ols_fit.cov_params()
# coefs_se = np.sqrt(np.diag(HCV_coefs))

