#!/usr/bin/env python
# coding: utf-8

# # Debiased ML for Partially Linear Model in Python

# ## Double/Debiased Machine Learning for the Partially Linear Regression Model.
# 
# This is a simple implementation of Debiased Machine Learning for the Partially Linear Regression Model.
# 
# Reference: 
# 
# https://arxiv.org/abs/1608.00060
# 
# 
# https://www.amazon.com/Business-Data-Science-Combining-Accelerate/dp/1260452778
# 
# The code is based on the book.

# In[1]:


# Import relevant packages
import pandas as pd
import numpy as np
import pyreadr
import os
from urllib.request import urlopen
from sklearn import preprocessing
import patsy

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

import hdmpy
import numpy as np
import random
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import itertools
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_categorical_dtype
from itertools import compress
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.feature_selection import SelectFromModel
from statsmodels.tools import add_constant
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


# ## DML algorithm
# 
# Here we perform estimation and inference of predictive coefficient $\alpha$ in the partially linear statistical model, 
# 
# $$
# \begin{align}
# Y = D\alpha + g(X) + U, \quad E (U | D, X) = 0. 
# \end{align}
# $$
# 
# For $\tilde Y = Y- E(Y|X)$ and $\tilde D= D- E(D|X)$, we can write
# 
# $$
# \begin{align}
# \tilde Y = \alpha \tilde D + U, \quad E (U |\tilde D) =0.
# \end{align}
# $$
# 
# Parameter $\alpha$ is then estimated using cross-fitting approach to obtain the residuals $\tilde D$ and $\tilde Y$.
# The algorithm comsumes $Y, D, X$, and machine learning methods for learning the residuals $\tilde Y$ and $\tilde D$, where
# the residuals are obtained by cross-validation (cross-fitting).
# 
# The statistical parameter $\alpha$ has a causal intertpreation of being the effect of $D$ on $Y$ in the causal DAG 
# 
# $$
# \begin{align}
# D\to Y, \quad X\to (D,Y)
# \end{align}
# $$
# 
# or the counterfactual outcome model with conditionally exogenous (conditionally random) assignment of treatment $D$ given $X$:
# 
# $$
# \begin{align}
# Y(d) = d\alpha + g(X) + U(d),\quad  U(d) \text{ indep } D |X, \quad Y = Y(D), \quad U = U(D).
# \end{align}
# $$
# 

# ### Clases needed for regression

# In[2]:


class standard_skl_model:
    
    def __init__(self, model ):
        self.model = model
       
    def fit( self, X, Y ):
        
        # Standarization of X and Y
        self.scaler_X = StandardScaler()
        self.scaler_X.fit( X )
        std_X = self.scaler_X.transform( X )
                
        self.model.fit( std_X , Y )
                
        return self
    
    def predict( self , X ):
        
        self.scaler_X = StandardScaler()
        self.scaler_X.fit( X )
        std_X = self.scaler_X.transform( X )
        
        prediction = self.model.predict( std_X )
        
        return prediction


# In[3]:


class rlasso_sklearn:
    
    def __init__(self, post ):
        self.post = post
       
    def fit( self, X, Y ):
        
        # Standarization of X and Y
        self.rlasso_model = hdmpy.rlasso( X , Y , post = self.post )                
        return self
    
    def predict( self , X ):
        
        beta = self.rlasso_model.est['coefficients'].to_numpy()
        prediction = ( add_constant( X ) @ beta ).flatten()
                
        return prediction


# ### Previous Function

# In[4]:


def DML2_for_PLM(x, d, y, dreg, yreg, nfold = 2 ):
    
    kf = KFold(n_splits = nfold, shuffle=True) #Here we use kfold to generate kfolds
    I = np.arange(0, len(d)) #To have a id vector from data
    train_id, test_id =  [], [] #arrays to store kfold's ids

    #generate and store kfold's id
    for kfold_index in kf.split(I):
        train_id.append(kfold_index[0])
        test_id.append(kfold_index[1])
    
    # Create array to save errors 
    dtil = np.zeros( len(d) ).reshape( len(d) , 1 )
    ytil = np.zeros( len(d) ).reshape( len(d) , 1 )
    
    # loop to save results
    for b in range(0,len(train_id)):

        # Lasso regression, excluding folds selected 
        dfit = dreg(x[train_id[b],], d[train_id[b],])
        yfit = yreg(x[train_id[b],], y[train_id[b],])

        # predict estimates using the 
        dhat = dfit.predict( x[test_id[b],] )
        yhat = yfit.predict( x[test_id[b],] )

        # save errors  
        dtil[test_id[b]] =  d[test_id[b],] - dhat.reshape( -1 , 1 )
        ytil[test_id[b]] = y[test_id[b],] - yhat.reshape( -1 , 1 )
        print(b, " ")
    
    # Create dataframe 
    data_2 = pd.DataFrame(np.concatenate( ( ytil, dtil ), axis = 1), columns = ['ytil','dtil' ])
   
    # OLS clustering at the County level
    model = "ytil ~ dtil"
    baseline_ols = smf.ols( model , data = data_2 ).fit().get_robustcov_results(cov_type = "HC3")
    coef_est = baseline_ols.summary2().tables[ 1 ][ 'Coef.' ][ 'dtil' ]
    se = baseline_ols.summary2().tables[ 1 ][ 'Std.Err.' ][ 'dtil' ]
    
    Final_result = { 'coef_est' : coef_est , 'se' : se , 'dtil' : dtil , 'ytil' : ytil }

    print( f"\n Coefficient (se) = {coef_est} ({se})" )
    
    return Final_result
    


# In[5]:


link="https://raw.githubusercontent.com/d2cml-ai/14.388_py/main/data/GrowthData.RData"
response = urlopen(link)
content = response.read()
fhandle = open( 'GrowthData.RData', 'wb')
fhandle.write(content)
fhandle.close()
result = pyreadr.read_r("GrowthData.RData")
os.remove("GrowthData.RData")

# Extracting the data frame from rdata_read
GrowthData = result[ 'GrowthData' ]
n = GrowthData.shape[0]


# In[6]:


y = GrowthData.iloc[ : , 0 ].to_numpy().reshape( GrowthData.shape[0] , 1 )
d = GrowthData.iloc[ : , 2].to_numpy().reshape( GrowthData.shape[0] , 1 )
x = GrowthData.iloc[ : , 3:].to_numpy()


# In[7]:


print( f'\n length of y is \n {y.size}' )
print( f'\n num features x is \n {x.shape[ 1 ]}' )

print( "\n Naive OLS that uses all features w/o cross-fitting \n" )
lres = sm.OLS( y , add_constant(np.concatenate( ( d , x ) , axis = 1 ) )  ).fit().summary2().tables[ 1 ].iloc[ 1, 0:2 ]
print( f'\n coef (se) = {lres[ 0 ]} ({lres[ 1 ]})' )




#DML with OLS
print( "\n DML with OLS w/o feature selection \n" )

def dreg(x,d):
    result = standard_skl_model( linear_model.Lasso( alpha = 0 , random_state = 0 )).fit( x, d )
    return result

def yreg(x,y):
    result = standard_skl_model( linear_model.Lasso( alpha = 0 ,  random_state = 0 ) ).fit( x, y )
    return result

DML2_ols = DML2_for_PLM(x, d, y, dreg, yreg, 10 )




# DML with LASSO
print( "\n DML with Lasso \n" )
def dreg(x,d):
    result = rlasso_sklearn( post = False ).fit( x , d )
    return result

def yreg(x,y):
    result = rlasso_sklearn( post = False ).fit( x , y )
    return result

DML2_lasso = DML2_for_PLM( x , d , y , dreg , yreg , 10 )




print( "\n DML with Random Forest \n" )

#DML with cross-validated Lasso:
def dreg( x , d ):
    result = RandomForestRegressor( random_state = 0 , n_estimators = 500 , max_features = 20 , n_jobs = 4 , min_samples_leaf = 5 ).fit( x, d )
    return result

def yreg( x , y ):
    result = RandomForestRegressor( random_state = 0 , n_estimators = 500 , max_features = 20 , n_jobs = 4 , min_samples_leaf = 5 ).fit( x, y )
    return result

DML2_RF = DML2_for_PLM( x , d , y , dreg , yreg , 2 )   # set to 2 due to computation time



#DML with Lasso:
print( "\n DML with Lasso/Random Forest \n" )
def dreg( x , d ):
    result = rlasso_sklearn( post = False ).fit( x , d )
    return result

def yreg( x , y ):
    result = RandomForestRegressor( random_state = 0 , n_estimators = 500 , max_features = 20 , n_jobs = 4 , min_samples_leaf = 5 ).fit( x, y )
    return result

DML2_RF = DML2_for_PLM( x , d , y , dreg , yreg , 2 )   # set to 2 due to computation time


# In[8]:


mods = [ DML2_ols, DML2_lasso, DML2_RF ]
mods_name = ["OLS", "Lasso", 'RF']

def mdl( model , model_name ):
    
    RMSEY = np.sqrt( np.mean( model[ 'ytil' ] ) ** 2 ) # I have some doubts about these equations...we have to recheck
    RMSED = np.sqrt( np.mean( model[ 'dtil' ] ) ** 2 ) # I have some doubts about these equations...we have to recheck
    
    result = pd.DataFrame( { model_name : [ RMSEY , RMSED ]} , index = [ 'RMSEY' , 'RMSED' ])
    return result

RES = [ mdl( model , name ) for model, name in zip( mods , mods_name ) ]


# In[9]:


pr_Res = pd.concat( RES, axis = 1)


# In[10]:


pr_Res.round( 7 )

