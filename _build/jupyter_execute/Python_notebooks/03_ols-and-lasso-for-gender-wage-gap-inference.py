#!/usr/bin/env python
# coding: utf-8

# * Python code replication of: " https://www.kaggle.com/janniskueck/pm1-notebook-inference "
# * Created by: Anzony Quispe & Alexander Quispe

# This notebook contains an example for teaching.

# # An inferential problem: The Gender Wage Gap

# In the previous lab, we already analyzed data from the March Supplement of the U.S. Current Population Survey (2015) and answered the question how to use job-relevant characteristics, such as education and experience, to best predict wages. Now, we focus on the following inference question:
# 
# What is the difference in predicted wages between men and women with the same job-relevant characteristics?
# 
# Thus, we analyze if there is a difference in the payment of men and women (*gender wage gap*). The gender wage gap may partly reflect *discrimination* against women in the labor market or may partly reflect a *selection effect*, namely that women are relatively more likely to take on occupations that pay somewhat less (for example, school teaching).

# To investigate the gender wage gap, we consider the following log-linear regression model
# 
# \begin{align}
# \log(Y) &= \beta'X + \epsilon\\
# &= \beta_1 D  + \beta_2' W + \epsilon,
# \end{align}
# 
# where $D$ is the indicator of being female ($1$ if female and $0$ otherwise) and the
# $W$'s are controls explaining variation in wages. Considering transformed wages by the logarithm, we are analyzing the relative difference in the payment of men and women.

# ## Data analysis

# We consider the same subsample of the U.S. Current Population Survey (2015) as in the previous lab. Let us load the data set.

# In[1]:


import pandas as pd
import numpy as np
import pyreadr
import math


# In[2]:


rdata_read = pyreadr.read_r("../data/wage2015_subsample_inference.Rdata")

# Extracting the data frame from rdata_read
data = rdata_read[ 'data' ]

data.shape


# To start our (causal) analysis, we compare the sample means given gender:

# In[3]:


Z = data[ ["lwage","sex","shs","hsg","scl","clg","ad","ne","mw","so","we","exp1"] ]

data_female = data[data[ 'sex' ] == 1 ]
Z_female = data_female[ ["lwage","sex","shs","hsg","scl","clg","ad","ne","mw","so","we","exp1"] ]

data_male = data[ data[ 'sex' ] == 0 ]
Z_male = data_male[ [ "lwage","sex","shs","hsg","scl","clg","ad","ne","mw","so","we","exp1" ] ]


table = np.zeros( (12, 3) )
table[:, 0] = Z.mean().values
table[:, 1] = Z_male.mean().values
table[:, 2] = Z_female.mean().values
table_pandas = pd.DataFrame( table, columns = [ 'All', 'Men', 'Women'])
table_pandas.index = ["Log Wage","Sex","Less then High School","High School Graduate","Some College","Gollage Graduate","Advanced Degree", "Northeast","Midwest","South","West","Experience"]
table_html = table_pandas.to_html()

table_pandas


# In[4]:


print( table_html )


# In particular, the table above shows that the difference in average *logwage* between men and women is equal to $0,038$

# In[5]:


data_female['lwage'].mean() - data_male['lwage'].mean()


# Thus, the unconditional gender wage gap is about $3,8$\% for the group of never married workers (women get paid less on average in our sample). We also observe that never married working women are relatively more educated than working men and have lower working experience.

# This unconditional (predictive) effect of gender equals the coefficient $\beta$ in the univariate ols regression of $Y$ on $D$:
# 
# $$
# \begin{align}
# \log(Y) &=\beta D + \epsilon.
# \end{align}
# $$

# We verify this by running an ols regression in R.

# In[6]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[7]:


nocontrol_model = smf.ols( formula = 'lwage ~ sex', data = data )
nocontrol_est = nocontrol_model.fit().summary2().tables[1]['Coef.']['sex']
HCV_coefs = nocontrol_model.fit().cov_HC0
nocontrol_se = np.power( HCV_coefs.diagonal() , 0.5)[1]

# print unconditional effect of gender and the corresponding standard error
print( f'The estimated gender coefficient is {nocontrol_est} and the corresponding robust standard error is {nocontrol_se}' )


# Note that the standard error is computed with the *R* package *sandwich* to be robust to heteroskedasticity. 
# 

# Next, we run an ols regression of $Y$ on $(D,W)$ to control for the effect of covariates summarized in $W$:
# 
# \begin{align}
# \log(Y) &=\beta_1 D  + \beta_2' W + \epsilon.
# \end{align}
# 
# Here, we are considering the flexible model from the previous lab. Hence, $W$ controls for experience, education, region, and occupation and industry indicators plus transformations and two-way interactions.

# Let us run the ols regression with controls.

# ## Ols regression with controls

# In[8]:


flex = 'lwage ~ sex + (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we)'

# The smf api replicates R script when it transform data
control_model = smf.ols( formula = flex, data = data )
control_est = control_model.fit().summary2().tables[1]['Coef.']['sex']

print(control_model.fit().summary2().tables[1])
print( f"Coefficient for OLS with controls {control_est}" )

HCV_coefs = control_model.fit().cov_HC0
control_se = np.power( HCV_coefs.diagonal() , 0.5)[1]


# The estimated regression coefficient $\beta_1\approx-0.0696$ measures how our linear prediction of wage changes if we set the gender variable $D$ from 0 to 1, holding the controls $W$ fixed.
# We can call this the *predictive effect* (PE), as it measures the impact of a variable on the prediction we make. Overall, we see that the unconditional wage gap of size $4$\% for women increases to about $7$\% after controlling for worker characteristics.  
# 

# Next, we are using the Frisch-Waugh-Lovell theorem from the lecture partialling-out the linear effect of the controls via ols.

# ## Partialling-Out using ols

# In[9]:


# models
# model for Y
flex_y = 'lwage ~  (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we)'
# model for D
flex_d = 'sex ~ (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we)' 

# partialling-out the linear effect of W from Y
t_Y = smf.ols( formula = flex_y , data = data ).fit().resid

# partialling-out the linear effect of W from D
t_D = smf.ols( formula = flex_d , data = data ).fit().resid

data_res = pd.DataFrame( np.vstack(( t_Y.values , t_D.values )).T , columns = [ 't_Y', 't_D' ] )
# regression of Y on D after partialling-out the effect of W
partial_fit =  smf.ols( formula = 't_Y ~ t_D' , data = data_res ).fit()
partial_est = partial_fit.summary2().tables[1]['Coef.']['t_D']

print("Coefficient for D via partialling-out", partial_est)

# standard error
HCV_coefs = partial_fit.cov_HC0
partial_se = np.power( HCV_coefs.diagonal() , 0.5)[1]

# confidence interval
partial_fit.conf_int( alpha=0.05 ).iloc[1, :]


# Again, the estimated coefficient measures the linear predictive effect (PE) of $D$ on $Y$ after taking out the linear effect of $W$ on both of these variables. This coefficient equals the estimated coefficient from the ols regression with controls.

# We know that the partialling-out approach works well when the dimension of $W$ is low
# in relation to the sample size $n$. When the dimension of $W$ is relatively high, we need to use variable selection
# or penalization for regularization purposes. 
# 
# In the following, we illustrate the partialling-out approach using lasso instead of ols. 

# ## Partialling-Out using lasso

# In[10]:


# models
# model for Y
flex_y = 'lwage ~  (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we)'

# model for D
flex_d = 'sex ~ (exp1+exp2+exp3+exp4)*(shs+hsg+scl+clg+occ2+ind2+mw+so+we)'


# In[11]:


# With Sklearn

from sklearn import linear_model
# flex_y
lasso_model = linear_model.Lasso( alpha = 0.1 )

flex_y_covariables = smf.ols(formula = flex_y, data = data)
Y_lasso_fitted = lasso_model.fit( flex_y_covariables.exog, data[[ 'lwage' ]] ).predict( flex_y_covariables.exog )
t_Y = data[[ 'lwage' ]] - Y_lasso_fitted.reshape( Y_lasso_fitted.size, 1)

# extraflex_d
flex_d_covariables = smf.ols( flex_d, data=data)
D_lasso_fitted = lasso_model.fit( flex_d_covariables.exog, data[[ 'sex' ]] ).predict( flex_d_covariables.exog )
t_D = data[[ 'sex' ]] - D_lasso_fitted.reshape( D_lasso_fitted.size, 1)

data_res = pd.DataFrame( np.hstack(( t_Y , t_D )) , columns = [ 't_Y', 't_D' ] )

# regression of Y on D after partialling-out the effect of W
partial_lasso_fit = smf.ols( formula = 't_Y ~ t_D' , data = data_res ).fit()
partial_lasso_est = partial_lasso_fit.summary2().tables[1]['Coef.']['t_D']

print( f"Coefficient for D via partialling-out using lasso {partial_lasso_est}" )

# standard error
HCV_coefs = partial_lasso_fit.cov_HC0
partial_lasso_se = np.power( HCV_coefs.diagonal() , 0.5)[1]


# Using lasso for partialling-out here provides similar results as using ols.

# Next, we summarize the results.

# ## Summarize the results

# In[12]:


table2 = np.zeros( (4, 2) )

table2[0,0] = nocontrol_est  
table2[0,1] = nocontrol_se   
table2[1,0] = control_est
table2[1,1] = control_se    
table2[2,0] = partial_est  
table2[2,1] = partial_se  
table2[3,0] =  partial_lasso_est
table2[3,1] = partial_lasso_se 

table2_pandas = pd.DataFrame( table2, columns = [ "Estimate","Std. Error" ])
table2_pandas.index = [ "Without controls", "full reg", "partial reg", "partial reg via lasso" ]
table2_html = table2_pandas.to_html()
table2_pandas


# It it worth to notice that controlling for worker characteristics increases the gender wage gap from less that 4\% to 7\%. The controls we used in our analysis include 5 educational attainment indicators (less than high school graduates, high school graduates, some college, college graduate, and advanced degree), 4 region indicators (midwest, south, west, and northeast);  a quartic term (first, second, third, and fourth power) in experience and 22 occupation and 23 industry indicators.
# 
# Keep in mind that the predictive effect (PE) does not only measures discrimination (causal effect of being female), it also may reflect
# selection effects of unobserved differences in covariates between men and women in our sample.
# 

# Next we try "extra" flexible model, where we take interactions of all controls, giving us about 1000 controls.

# ## "Extra" flexible model

# In[13]:


extraflex = 'lwage ~ sex + (exp1+exp2+exp3+exp4+shs+hsg+scl+clg+occ2+ind2+mw+so+we)**2'

control_fit = smf.ols( formula = extraflex, data=data).fit()

#summary( control_fit )
control_est = control_fit.summary2().tables[1]['Coef.']['sex']

print( f"Number of Extra-Flex Controls {control_fit.summary2().tables[1].shape[0]-1} \nCoefficient for OLS with extra flex controls {control_est}" )

# standard error
HCV_coefs = control_fit.cov_HC0

n= len(data[ 'wage' ])

p = len(control_fit.summary2().tables[1]['Coef.'])

control_se = control_fit.summary2().tables[1]['Std.Err.']['sex']*math.sqrt(n/(n-p))
control_se
# crude adjustment for the effect of dimensionality on OLS standard errors, motivated by Cattaneo, Jannson, and Newey (2018)

# for really correct way of doing this, we need to implement Cattaneo, Jannson, and Newey (2018)'s procedure.


# ## Laso "Extra" Flexible model

# In[14]:


# models
# model for Y
extraflex_y = 'lwage ~  (exp1+exp2+exp3+exp4+shs+hsg+scl+clg+occ2+ind2+mw+so+we)**2'

# model for 
extraflex_d = 'sex ~ (exp1+exp2+exp3+exp4+shs+hsg+scl+clg+occ2+ind2+mw+so+we)**2'

# extraflex_y
lasso_model = linear_model.Lasso( alpha = 0.1  )

extraflex_y_covariables = smf.ols(formula = extraflex_y, data = data)

Y_lasso_fitted = lasso_model.fit( extraflex_y_covariables.exog, data[[ 'lwage' ]] ).predict( extraflex_y_covariables.exog )

t_Y = data[[ 'lwage' ]] - Y_lasso_fitted.reshape( Y_lasso_fitted.size, 1)

# extraflex_d
extraflex_d_covariables = smf.ols( extraflex_d, data=data)

D_lasso_fitted = lasso_model.fit( extraflex_d_covariables.exog, data[[ 'sex' ]] ).predict( extraflex_d_covariables.exog )

t_D = data[[ 'sex' ]] - D_lasso_fitted.reshape( D_lasso_fitted.size, 1)

data_res = pd.DataFrame( np.hstack(( t_Y , t_D )) , columns = [ 't_Y', 't_D' ] )

# regression of Y on D after partialling-out the effect of W
partial_lasso_fit = smf.ols( formula = 't_Y ~ t_D' , data = data_res ).fit()
partial_lasso_est = partial_lasso_fit.summary2().tables[1]['Coef.']['t_D']

print( f"Coefficient for D via partialling-out using lasso {partial_lasso_est}" )

# standard error
HCV_coefs = partial_lasso_fit.cov_HC0
partial_lasso_se = np.power( HCV_coefs.diagonal() , 0.5)[1]


# ## Summarize the results

# In[15]:


table3 = np.zeros( ( 2, 2 ) )

table3[0,0] = control_est
table3[0,1] = control_se    
table3[1,0] =  partial_lasso_est
table3[1,1] = partial_lasso_se 

table3_pandas = pd.DataFrame( table3, columns = [ "Estimate","Std. Error" ])
table3_pandas.index = [ "full reg","partial reg via lasso" ]
table3_pandas.round(8)


# In this case p/n = 20%, that is  p/n  is no longer small and we start seeing the differences between unregularized partialling out and regularized partialling out with lasso (double lasso). The results based on double lasso have rigorous guarantees in this non-small p/n regime under approximate sparsity. The results based on OLS still have guarantees in p/n< 1 regime under assumptions laid out in Cattaneo, Newey, and Jansson (2018), without approximate sparsity, although other regularity conditions are needed.
