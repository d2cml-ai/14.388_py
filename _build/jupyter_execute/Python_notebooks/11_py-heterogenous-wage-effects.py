#!/usr/bin/env python
# coding: utf-8

# # Heterogenous Wage Effects

# ## Application: Heterogeneous Effect of Gender on Wage Using Double Lasso
# 
#  We use US census data from the year 2012 to analyse the effect of gender and interaction effects of other variables with gender on wage jointly. The dependent variable is the logarithm of the wage, the target variable is *female* (in combination with other variables). All other variables denote some other socio-economic characteristics, e.g. marital status, education, and experience.  For a detailed description of the variables we refer to the help page.
# 
# 
# 
# This analysis allows a closer look how discrimination according to gender is related to other socio-economic variables.
# 
# 

# In[1]:


import hdmpy
import pyreadr
import patsy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import stats
import numpy as np
import scipy.linalg as sci_lag
import warnings
warnings.filterwarnings('ignore')


# In[2]:


cps2012_env = pyreadr.read_r("../data/cps2012.Rdata")
cps2012 = cps2012_env[ 'data' ]
cps2012.describe()


# In[3]:


formula_basic =  '''lnw ~ -1 + female + female:(widowed + divorced + separated + nevermarried +
hsd08 + hsd911 + hsg + cg + ad + mw + so + we + exp1 + exp2 + exp3) + +(widowed +
divorced + separated + nevermarried + hsd08 + hsd911 + hsg + cg + ad + mw + so +
we + exp1 + exp2 + exp3) ** 2'''

y, X = patsy.dmatrices(formula_basic, cps2012, return_type='dataframe')
X.shape[1]


# We have the same number of covariables.

# In[4]:


variance_cols = X.var().to_numpy()
X = X.iloc[ : ,  np.where( variance_cols != 0   )[0] ]

def demean(x):
    dif = x - np.mean( x )
    return dif 

X = X.apply( demean, axis = 0 )

index_gender = np.where( X.columns.str.contains('female'))[0]


# The parameter estimates for the target parameters, i.e. all coefficients related to gender (i.e. by interaction with other variables) are calculated and summarized by the following commands:

# In[5]:


effect_female = hdmpy.rlassoEffects( x = X , y = y , index = index_gender )


# In[6]:


result_coeff = pd.concat( [ effect_female.res.get( 'coefficients' ).rename(columns = { 0 : "Estimate." }) ,              effect_female.res.get( 'se' ).rename( columns = { 0 : "Std. Error" } ) ,              effect_female.res.get( 't' ).rename( columns = { 0 : "t value" } ) ,              effect_female.res.get( 'pval' ).rename( columns = { 0 : "Pr(>|t|)" } ) ] ,             axis = 1 )

print( result_coeff )


# In[7]:


t_value = stats.t.ppf(1-0.05, 29217 - 116 -1 )


# In[8]:


pointwise_CI = pd.DataFrame({ '5%' : result_coeff.iloc[ : , 0 ]                                      - result_coeff.iloc[ : , 1 ] * t_value ,                              '95%' : result_coeff.iloc[ : , 0 ]                              + result_coeff.iloc[ : , 1 ] * t_value })
pointwise_CI


# In[9]:


result_coeff = result_coeff.sort_values('Estimate.')

x = result_coeff.index

coef = result_coeff.iloc[ : , 0 ].to_numpy()

sd_error = ( result_coeff.iloc[ : , 1 ] * t_value ).to_numpy()

figure(figsize=(12, 6), dpi=80)

plt.errorbar( x = x , y = coef , yerr = sd_error , linestyle="None" , color = "black",               capsize = 3 , marker = "s" , markersize = 3 , mfc = "black" , mec = "black" )
plt.xticks(x, x, rotation=90)
plt.show()


# Now, we estimate and plot confident intervals, first "pointwise" and then the joint confidence intervals.

# In[10]:


effect_female.res.get( 'coefficients' ).iloc[ :, 0 ].to_list()


# In[11]:


def confint_joint_python( rlassomodel , level = 0.95 ):
    
    coef = rlassomodel.res['coefficients'].to_numpy().reshape( 1 , rlassomodel.res['coefficients'].to_numpy().size )
    se = rlassomodel.res['se'].to_numpy().reshape( 1 , rlassomodel.res['se'].to_numpy().size )
    
    e = rlassomodel.res['residuals']['e']
    v = rlassomodel.res['residuals']['v']
    
    n = e.shape[0]
    k = e.shape[1]
    
    ev = e.to_numpy() * v.to_numpy()
    Ev2 = (v ** 2).mean( axis = 0 ).to_numpy()
    
    
    Omegahat = np.zeros( ( k , k ) )
    
    
    for j in range( 0 , k ):
        for l in range( 0 , k ):
            Omegahat[ j , l ] = ( 1 / ( Ev2[ j ] * Ev2[ l ] ) ) * np.mean(ev[ : , j ] * ev[ : , l ] )
            Omegahat[ l , j ] = ( 1 / ( Ev2[ j ] * Ev2[ l ] ) ) * np.mean(ev[ : , j ] * ev[ : , l ] )
    
    var = np.diagonal( Omegahat )
    
    B = 500
    sim = np.zeros( B )
    
    for i in range( 0 , B ):
        beta_i = mvrnorm( mu =  np.repeat( 0, k ) , Sigma =  Omegahat / n )
        sim[ i ] = ( np.abs( beta_i/ ( var ** 0.5 ) ) ).max()
    
    t_stat = np.quantile( sim , level )
    
    low_num = int( np.round( ( 1 - level ) / 2, 2) * 100 )
    upper_num = int( np.round( ( 1 + level ) / 2, 2) * 100 )
    
    sd_tstat = t_stat * ( var ** 0.5 )
    
    low = coef - sd_tstat
    upper = coef + sd_tstat
    
    table = pd.DataFrame( { f'{low_num}%' : low.tolist()[0] , f'{upper_num}%' : upper.tolist()[0] ,                            'Coef.' : rlassomodel.res.get( 'coefficients' ).iloc[ :, 0 ].to_list() },                          index = rlassomodel.res.get( 'coefficients' ).index )
    
    return ( table, sd_tstat )


# In[12]:


def mvrnorm( mu , Sigma , n = 1 ):
    p = mu.size
    
    ev = np.linalg.eig( Sigma )[ 0 ]
    vectors = np.linalg.eig( Sigma )[ 1 ]
    
    X = np.random.normal(0, 1 ,  p * n ).reshape( n , p )
    
    diagonal = np.diag( np.maximum( ev, 0 ) ** 0.5 )
    
    Z = mu.reshape( p , 1 ) + vectors @ ( diagonal @ X.transpose())
    
    resultado = Z.transpose()
    
    return( resultado )


# Finally, we compare the pointwise confidence intervals to joint confidence intervals.

# In[13]:


from statsmodels.sandbox.stats.multicomp import multipletests
import scipy.stats as st


# In[14]:


joint_CI =  confint_joint_python( effect_female, level = 0.9 )[0]
size_err =  confint_joint_python( effect_female, level = 0.9 )[1]


# In[15]:


joint_CI


# In[16]:


x = joint_CI.index

coef = joint_CI.iloc[ : , 1 ].to_numpy()

sd_error = size_err

figure(figsize=(12, 6), dpi=80)

plt.errorbar( x = x , y = coef , yerr = sd_error , linestyle="None" , color = "black",               capsize = 3 , marker = "s" , markersize = 3 , mfc = "black" , mec = "black" )
plt.xticks(x, x, rotation=90)
plt.show()

