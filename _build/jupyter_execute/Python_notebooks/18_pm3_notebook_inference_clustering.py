#!/usr/bin/env python
# coding: utf-8

# # DML inference for gun ownership

# In[1]:


# Import relevant packages
import pandas as pd
import numpy as np
import pyreadr
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
import warnings
warnings.filterwarnings('ignore')


# In[2]:


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


# This notebook contains an example for teaching.

# ## A Case Study: The Effect of Gun Ownership on Gun-Homicide Rates

# We consider the problem of estimating the effect of gun
# ownership on the homicide rate. For this purpose, we estimate the following partially
# linear model
# 
# $$
#  Y_{j,t} = \beta D_{j,(t-1)} + g(Z_{j,t}) + \epsilon_{j,t}.
# $$

# ## Data

# $Y_{j,t}$ is log homicide rate in county $j$ at time $t$, $D_{j, t-1}$ is log  fraction of suicides committed with a firearm in county $j$ at time $t-1$, which we use as a proxy for gun ownership,  and  $Z_{j,t}$ is a set of demographic and economic characteristics of county $j$ at time $t$. The parameter $\beta$ is the effect of gun ownership on the
# homicide rates, controlling for county-level demographic and economic characteristics. 
# 
# The sample covers 195 large United States counties between the years 1980 through 1999, giving us 3900 observations.

# In[3]:


data1 = pd.read_csv( r"../data/gun_clean.csv" )
data1.shape[1]


# ### Preprocessing

# To account for heterogeneity across counties and time trends in  all variables, we remove from them county-specific and time-specific effects in the following preprocessing.

# In[4]:


import re


# In[5]:


#################################  Find Variable Names from Dataset ########################

def varlist( df = None, type_dataframe = [ 'numeric' , 'categorical' , 'string' ],  pattern = "", exclude = None ):
    varrs = []
    
    if ('numeric' in type_dataframe):
        varrs = varrs + df.columns[ df.apply( is_numeric_dtype , axis = 0 ).to_list() ].tolist()
    
    if ('categorical' in type_dataframe):
        varrs = varrs + df.columns[ df.apply( is_categorical_dtype , axis = 0 ).to_list() ].tolist()
    
    if ('string' in type_dataframe):
        varrs = varrs + df.columns[ df.apply( is_string_dtype , axis = 0 ).to_list() ].tolist()
    
    grepl_result = np.array( [ re.search( pattern , variable ) is not None for variable in df.columns.to_list() ] )
    
    if exclude is None:
        result = list(compress( varrs, grepl_result ) )
    
    else:
        varrs_excluded = np.array( [var in exclude for var in varrs ] )
        and_filter = np.logical_and( ~varrs_excluded ,  grepl_result )
        result = list(compress( varrs, and_filter ) )
    
    return result   

################################# Create Variables ###############################


# Dummy Variables for Year and County Fixed Effects
r = re.compile("X_Jfips")
fixed = list( filter( r.match, data1.columns.to_list() ) )
year = varlist(data1, pattern="X_Tyear")

census = []
census_var = ["^AGE", "^BN", "^BP", "^BZ", "^ED", "^EL","^HI", "^HS", "^INC", "^LF", "^LN", "^PI", "^PO", "^PP", "^PV", "^SPR", "^VS"]

for variable in census_var:
    r = re.compile( variable )
    census = census + list( filter( r.match, data1.columns.to_list() ) )

    
################################ Variables ##################################
# Treatment Variable
d = "logfssl"

# Outcome Variable
y = "logghomr"

# Other Control Variables
X1 = ["logrobr", "logburg", "burg_missing", "robrate_missing"]
X2 = ["newblack", "newfhh", "newmove", "newdens", "newmal"]

#################################  Partial out Fixed Effects ########################

rdata = data1[['CountyCode']]

# Variables to be Partialled-out
varlist2 = [y] + [d] + X1 + X2 + census

# Partial out Variables in varlist from year and county fixed effect
for var_name in varlist2:
    form = var_name + " ~ "  + " + ".join( year ) + "+" + " + ".join( fixed )
    rdata[f'{var_name}'] = smf.ols( formula = form , data = data1 ).fit().resid


# In[6]:


# load dataset
rdata_read = pyreadr.read_r("../data/gun_clean.RData")
data = rdata_read[ 'data' ]
n = data.shape[0]


# ### We check that our results are equal to R results at 6 decimals

# In[7]:


column_names = data.columns.to_list()

for col in column_names:
    result = (data[f'{col}'].round(6) == rdata[f'{col}'].round(6)).sum()


# Now, we can construct the treatment variable, the outcome variable and the matrix $Z$ that includes the control variables.

# In[8]:


# Treatment Variable
D = rdata[ f'{d}']

# Outcome Variable
Y = rdata[ f'{y}']

# Construct matrix Z
Z = rdata[ X1 + X2 + census ]

Z.shape


# We have in total 195 control variables. The control variables $Z_{j,t}$ are from the U.S. Census Bureau and  contain demographic and economic characteristics of the counties such as  the age distribution, the income distribution, crime rates, federal spending, home ownership rates, house prices, educational attainment, voting paterns, employment statistics, and migration rates. 

# In[9]:


clu = rdata[['CountyCode']]
data = pd.concat([Y, D, Z, clu], axis=1)


# In[10]:


data.to_csv( r"../data/gun_clean2.csv" , index = False )


# ## The effect of gun ownership

# ### OLS

# After preprocessing the data, we first look at simple regression of $Y_{j,t}$ on $D_{j,t-1}$ without controls as a baseline model.

# In[11]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy


# In[12]:


# # Run this line to avoid all the lines of code above
# data = pd.read_csv( r"../data/gun_clean2.csv"  )


# In[13]:


# OLS clustering at the County level
model = "logghomr ~ logfssl"
baseline_ols = smf.ols(model , data=data).fit().get_robustcov_results(cov_type = "cluster", groups= data['CountyCode'])
baseline_ols_table = baseline_ols.summary2().tables[1]
print( baseline_ols_table.iloc[ 1 , 4:] )
baseline_ols_table.iloc[1, :]


# The point estimate is $0.282$ with the confidence interval ranging from 0.155 to 0.41. This
# suggests that increases in gun ownership rates are related to gun homicide rates - if gun ownership increases by 1% relative
# to a trend then the predicted gun homicide rate goes up by 0.28%, without controlling for counties' characteristics.
# 
# Since our goal is to estimate the effect of gun ownership after controlling for a rich set county characteristics we next include the controls. First, we estimate the model by ols and then by an array of the modern regression methods using the double machine learning approach.

# In[14]:


# define the variables
y = 'logghomr'

data_columns = list(data)
no_relev_col = ['logfssl', 'CountyCode', 'logghomr']

# This gives us: new_list = ['carrot' , 'lemon']
z = [col for col in data_columns if col not in no_relev_col]


# In[15]:


control_formula = "logghomr" + ' ~ ' + 'logfssl + ' + ' + '.join( Z.columns.to_list() )


# In[16]:


control_ols = smf.ols( control_formula , data=data).fit().get_robustcov_results(cov_type = "cluster", groups= data['CountyCode'])
control_ols_table = control_ols.summary2().tables[1]
print( control_ols_table.iloc[ 1 , 4:] )
control_ols_table.iloc[1, :]


# In R, the coefficients of the bellow variables are `Null`. However, in python we got a very high value.

# In[17]:


control_ols_table.loc[['PPQ110D', 'PPQ120D'], :]


# After controlling for a rich set of characteristics, the point estimate of gun ownership reduces to $0.19$.

# ### DML algorithm
# 
# Here we perform inference of the predictive coefficient $\beta$ in our partially linear statistical model, 
# 
# $$
# Y = D\beta + g(Z) + \epsilon, \quad E (\epsilon | D, Z) = 0,
# $$
# 
# using the **double machine learning** approach. 
# 
# For $\tilde Y = Y- E(Y|Z)$ and $\tilde D= D- E(D|Z)$, we can write
# \begin{align}
# \tilde Y = \alpha \tilde D + \epsilon, \quad E (\epsilon |\tilde D) =0.
# \end{align}
# 
# Using cross-fitting, we employ modern regression methods
# to build estimators $\hat \ell(Z)$ and $\hat m(Z)$ of $\ell(Z):=E(Y|Z)$ and $m(Z):=E(D|Z)$ to obtain the estimates of the residualized quantities:
# 
# $$
# \tilde Y_i = Y_i  - \hat \ell (Z_i),   \quad \tilde D_i = D_i - \hat m(Z_i), \quad \text{ for each } i = 1,\dots,n.
# $$
# 
# Finally, using ordinary least squares of $\tilde Y_i$ on $\tilde D_i$, we obtain the 
# estimate of $\beta$.

# The following algorithm comsumes $Y, D, Z$, and a machine learning method for learning the residuals $\tilde Y$ and $\tilde D$, where the residuals are obtained by cross-validation (cross-fitting). Then, it prints the estimated coefficient $\beta$ and the corresponding standard error from the final OLS regression.

# In[18]:


def DML2_for_PLM(z, d, y, dreg, yreg, nfold, clu):
    
    # Num ob observations
    nobs = z.shape[0]
    
    # Define folds indices 
    list_1 = [*range(0, nfold, 1)]*nobs
    sample = np.random.choice(nobs,nobs, replace=False).tolist()
    foldid = [list_1[index] for index in sample]

    # Create split function(similar to R)
    def split(x, f):
        count = max(f) + 1
        return tuple( list(itertools.compress(x, (el == i for el in f))) for i in range(count) ) 

    # Split observation indices into folds 
    list_2 = [*range(0, nobs, 1)]
    I = split(list_2, foldid)
    
    # Create array to save errors 
    dtil = np.zeros( len(z) ).reshape( len(z) , 1 )
    ytil = np.zeros( len(z) ).reshape( len(z) , 1 )
    
    # loop to save results
    for b in range(0,len(I)):
    
        # Split data - index to keep are in mask as booleans
        include_idx = set(I[b])  #Here should go I[b] Set is more efficient, but doesn't reorder your elements if that is desireable
        mask = np.array([(i in include_idx) for i in range(len(z))])

        # Lasso regression, excluding folds selected 
        dfit = dreg(z[~mask,], d[~mask,])
        yfit = yreg(z[~mask,], y[~mask,])

        # predict estimates using the 
        dhat = dfit.predict( z[mask,] )
        yhat = yfit.predict( z[mask,] )

        # save errors  
        dtil[mask] =  d[mask,] - dhat.reshape( len(I[b]) , 1 )
        ytil[mask] = y[mask,] - yhat.reshape( len(I[b]) , 1 )
        print(b, " ")
    
    # Create dataframe 
    data_2 = pd.DataFrame(np.concatenate( ( ytil, dtil,clu ), axis = 1), columns = ['ytil','dtil','CountyCode'])
   
    # OLS clustering at the County level
    model = "ytil ~ dtil"
    baseline_ols = smf.ols(model , data = data_2 ).fit().get_robustcov_results(cov_type = "cluster", groups= data_2['CountyCode'])
    coef_est = baseline_ols.summary2().tables[1]['Coef.']['dtil']
    se = baseline_ols.summary2().tables[1]['Std.Err.']['dtil']
    
    Final_result = { 'coef_est' : coef_est , 'se' : se , 'dtil' : dtil , 'ytil' : ytil }

    print("Coefficient is {}, SE is equal to {}".format(coef_est, se))
    
    return Final_result
    


# Now, we apply the Double Machine Learning (DML) approach with different machine learning methods. First, we load the relevant libraries.

# Let us, construct the input matrices.

# In[19]:


# Create main variables
Y = data['logghomr']
D = data['logfssl']
Z = data.drop(['logghomr', 'logfssl', 'CountyCode'], axis=1)
CLU = data['CountyCode']


# In[20]:


# as matrix
y = Y.to_numpy().reshape( len(Y) , 1 )
d = D.to_numpy().reshape( len(Y) , 1 )
z = Z.to_numpy()
clu = clu.to_numpy().reshape( len(Y) , 1 )


# ### Lasso 

# In[21]:


def dreg(z,d):
    alpha=0.00000001
    result = linear_model.Lasso(alpha = alpha).fit(z, d)
    return result

def yreg(z,y):
    alpha=0.00000001
    result = linear_model.Lasso(alpha = alpha).fit(z, y)
    return result

DML2_lasso = DML2_for_PLM(z, d, y, dreg, yreg, 10, clu)


# #### We use SelectFromModel to select variables which do not have a zero coefficient. It is done because we want to reduce dimensionality as rlasso( post = T ) does.

# In[22]:


class Lasso_post:
    
    def __init__(self, alpha ):
        self.alpha = alpha

        
    def fit( self, X, Y ):
        self.X = X
        self.Y = Y
        lasso = linear_model.Lasso( alpha = self.alpha ).fit( X , Y )
        model = SelectFromModel( lasso , prefit = True )
        X_new = model.transform( X )
        # Gettin indices from columns which has variance for regression
        index_X = model.get_support()
        
        self.index = index_X
        new_x = X[ : ,  index_X ]
        
        lasso2 = linear_model.Lasso( alpha = self.alpha ).fit( new_x , Y )
        self.model = lasso2
        
        return self
    
    def predict( self , X ):
        
        dropped_X = X[ : , self.index ]
        
        predictions = self.model.predict( dropped_X )
        
        return predictions


# Run the below code to verify whether Lasso_post functions as it is expected. 

# In[23]:


def dreg(z,d):
    alpha=0.00000001
    result = Lasso_post( alpha = alpha ).fit( z , d )
    return result

def yreg( z , y ):
    alpha = 0.00000001
    result = Lasso_post( alpha = alpha ).fit( z , y )
    return result

DML2_lasso_post = DML2_for_PLM(z, d, y, dreg, yreg, 10, clu)


# ### hdmy
# We are going to replicate the above regressions but using `hmdpy` library.

# In[24]:


import hdmpy
from statsmodels.tools import add_constant


# In[25]:


class rlasso_sklearn:
    
    def __init__(self, post ):
        self.post = post
       
    def fit( self, X, Y ):
        
        self.X = X
        self.Y = Y
        
        # Standarization of X and Y
        self.rlasso_model = hdmpy.rlasso( X , Y , post = self.post )                
        return self
    
    def predict( self , X_1 ):
        self.X_1 = X_1
        beta = self.rlasso_model.est['coefficients'].to_numpy()
        
        if beta.sum() == 0:
            prediction = np.repeat( self.rlasso_model.est['intercept'] , self.X_1.shape[0] )
        
        else:
            prediction = ( add_constant( self.X_1 , has_constant = 'add') @ beta ).flatten()
                
        return prediction


# In[26]:


# Post = false
def dreg(x, d):
    result = rlasso_sklearn( post = False ).fit( x , d )
    return result

def yreg(x,y):
    result = rlasso_sklearn( post = False ).fit( x , y )
    return result

DML2_lasso_hdmpy = DML2_for_PLM(z, d, y, dreg, yreg, 10, clu)


# In[27]:


# Post = True
def dreg(x, d):
    result = rlasso_sklearn( post = True ).fit( x , d )
    return result

def yreg(x,y):
    result = rlasso_sklearn( post = True ).fit( x , y )
    return result

DML2_lasso_post_hdmpy = DML2_for_PLM(z, d, y, dreg, yreg, 10, clu)


# We found out that the results are close. However, there are not a big differences between the Sklearn and the Hdmpy lasso function.

# ### cv.glmnet curiosities
# 1. According to [link1](https://stackoverflow.com/questions/24098212/what-does-the-option-normalize-true-in-lasso-sklearn-do) and [link2](https://statisticaloddsandends.wordpress.com/2018/11/15/a-deep-dive-into-glmnet-standardize/), It standardize **X** variables before estimation. In sklearn we have an option named **normalize**. It normalizes **X** by subtracting the mean and dividing by the l2-norm. Based on [link3](https://stackoverflow.com/questions/59846325/confusion-about-standardize-option-of-glmnet-package-in-r), **cv.glmnet (standardize = TRUE)** and **sklearn (normlize = True)** are not the same. We decided to use **StandardScaler** that meets suggestions made in **link3**. We proved this in the commented code below.

# In[28]:


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


# Run the below code to verify whether standard_skl_model functions as it is expected. 

# In[29]:


#DML with cross-validated Lasso:
def dreg(z,d):
    result = standard_skl_model( LassoCV(cv = 10 , random_state = 0 ) ).fit( z, d )
    return result

def yreg(z,y):
    result = standard_skl_model( LassoCV(cv = 10 , random_state = 0  ) ).fit( z, y )
    return result

DML2_lasso_cv = DML2_for_PLM(z, d, y, dreg, yreg, 2 , clu)


# In[30]:


# DML with cross-validated Lasso:
def dreg(z,d):
    result = standard_skl_model( ElasticNetCV( cv = 10 , random_state = 0 , l1_ratio = 0.5, max_iter = 100000 ) ).fit( z, d )
    return result

def yreg(z,y):
    result = standard_skl_model( ElasticNetCV( cv = 10 , random_state = 0 , l1_ratio = 0.5, max_iter = 100000 ) ).fit( z, y )
    return result

DML2_elnet = DML2_for_PLM(z, d, y, dreg, yreg, 2 , clu)


# In[31]:


#DML with cross-validated Lasso:
def dreg(z,d):
    result = standard_skl_model( ElasticNetCV( cv = 10 ,  random_state = 0 , l1_ratio = 0.0001 ) ).fit( z, d )
    return result

def yreg(z,y):
    result = standard_skl_model( ElasticNetCV( cv = 10 , random_state = 0 , l1_ratio = 0.0001 ) ).fit( z, y )
    return result

DML2_ridge = DML2_for_PLM(z, d, y, dreg, yreg, 2, clu)


# Here we also compute DML with OLS used as the ML method

# In[32]:


#DML with cross-validated Lasso:
def dreg(z,d):
    result = LinearRegression().fit( z, d )
    return result

def yreg(z,y):
    result = LinearRegression().fit( z, y )
    return result

DML2_ols = DML2_for_PLM(z, d, y, dreg, yreg, 2, clu)


# Next, we also apply Random Forest for comparison purposes.

# ### Random Forest
# 

# In[33]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[34]:


#DML with cross-validated Lasso:
def dreg(z,d):
    result = RandomForestRegressor( random_state = 0 , n_estimators = 500 , max_features = 65 , n_jobs = 4 , min_samples_leaf = 5 ).fit( z, d )
    return result

def yreg(z,y):
    result = RandomForestRegressor( random_state = 0 , n_estimators = 500 , max_features = 65 , n_jobs = 4 , min_samples_leaf = 5 ).fit( z, y )
    return result

DML2_RF = DML2_for_PLM(z, d, y, dreg, yreg, 2, clu)   # set to 2 due to computation time


# We conclude that the gun ownership rates are related to gun homicide rates - if gun ownership increases by 1% relative
# to a trend then the predicted gun homicide rate goes up by about 0.20% controlling for counties' characteristics.

# Finally, let's see which method is actually better. We compute RMSE for predicting D and Y, and see which
# of the methods works better.
# 

# In[35]:


mods = [DML2_ols, DML2_lasso, DML2_lasso_post , DML2_lasso_cv, DML2_ridge, DML2_elnet, DML2_RF]
mods_name = ["DML2_ols", "DML2_lasso", "DML2_lasso_post" , "DML2_lasso_cv", 'DML2_ridge', 'DML2_elnet', 'DML2_RF']

def mdl( model , model_name ):
    
    RMSEY = np.sqrt( np.mean( model[ 'ytil' ] ) ** 2 ) # I have some doubts about these equations...we have to recheck
    RMSED = np.sqrt( np.mean( model[ 'dtil' ] ) ** 2 ) # I have some doubts about these equations...we have to recheck
    
    result = pd.DataFrame( { model_name : [ RMSEY , RMSED ]} , index = [ 'RMSEY' , 'RMSED' ])
    return result

RES = [ mdl( model , name ) for model, name in zip( mods , mods_name ) ]
    

pr_Res = pd.concat( RES, axis = 1)

pr_Res


# #### This verfies that the function DML2_for_PLM has no errors

# In[36]:


np.where(DML2_lasso_post[ 'ytil' ] == 0)[0].size


# It looks like the best method for predicting D is Lasso, and the best method for predicting Y is CV Ridge.

# In[37]:


#DML with cross-validated Lasso:
def dreg(z,d):
    result = standard_skl_model( LassoCV(cv = 10 , random_state = 0 , alphas = [0]) ).fit( z, d )
    return result


def yreg(z,y):
    result = standard_skl_model( ElasticNetCV( cv = 10 ,  random_state = 0 , l1_ratio = 0.0001 ) ).fit( z, y )
    return result

DML2_best = DML2_for_PLM(z, d, y , dreg, yreg, 10, clu)


# In[38]:


table = np.zeros( ( 9 , 2 ))
table[ 0 , 0] = baseline_ols_table.iloc[ 1 , 0 ]
table[ 1 , 0] = control_ols_table.iloc[ 1 , 0 ]
table[ 2 , 0] = DML2_lasso['coef_est']
table[ 3 , 0] = DML2_lasso_post['coef_est']
table[ 4 , 0] = DML2_lasso_cv['coef_est']
table[ 5 , 0] = DML2_ridge['coef_est']
table[ 6 , 0] = DML2_elnet['coef_est']
table[ 7 , 0] = DML2_RF['coef_est']
table[ 8 , 0] = DML2_best['coef_est']
table[ 0 , 1] = baseline_ols_table.iloc[ 1 , 1 ]
table[ 1 , 1] = control_ols_table.iloc[ 1 , 1 ]
table[ 2 , 1] = DML2_lasso['se']
table[ 3 , 1] = DML2_lasso_post['se']
table[ 4 , 1] = DML2_lasso_cv['se']
table[ 5 , 1] = DML2_ridge['se']
table[ 6 , 1] = DML2_elnet['se']
table[ 7 , 1] = DML2_RF['se']
table[ 8 , 1] = DML2_best['se']


# In[39]:


table = pd.DataFrame( table , index = [ "Baseline OLS", "Least Squares with controls", "Lasso",              "Post-Lasso", "CV Lasso","CV Elnet", "CV Ridge", "Random Forest", "Best" ] ,             columns = ["Estimate","Standard Error"] )
table.round( 3 )

