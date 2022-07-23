#!/usr/bin/env python
# coding: utf-8

# # DML inference using NN for gun ownership

# ## The Effect of Gun Ownership on Gun-Homicide Rates using DML for neural nets

# In this lab, we estimate the effect of gun ownership on the homicide rate by a neural network.

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


# In[3]:


data1 = pd.read_csv( r"../data/gun_clean.csv" )
data1.shape[1]


# ## Preprocessing

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
    form = var_name + " ~ " + "+" + " + ".join( year ) + "+" + " + ".join( fixed )
    rdata[f'{var_name}'] = smf.ols( formula = form , data = data1 ).fit().resid


# ## DML for neural nets
# 

# The following algorithm comsumes $Y$,$D$ and $Z$, and learns the residuals $\tilde{Y}$ and $\tilde{D}$ via a neural network, where the residuals are obtained by cross-validation (cross-fitting). Then, it prints the estimated coefficient β and the clustered standard error from the final OLS regression.

# In[11]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[12]:


def DML2_for_NN(z, d, y, nfold, clu, num_epochs, batch_size):
    
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
    
    # loop to save results
    for b in range(0,len(I)):
    
        # Split data - index to keep are in mask as booleans
        include_idx = set(I[b])  #Here should go I[b] Set is more efficient, but doesn't reorder your elements if that is desireable
        mask = np.array([(i in include_idx) for i in range(len(z))])

        # Normalize the data
        scaler = StandardScaler()
        
        scaler.fit( z[~mask,] )
        z[~mask,] = scaler.transform( z[~mask,] )

        scaler.fit( z[mask,] )
        z[mask,] = scaler.transform( z[mask,] )
        
        # building the model
        # define the keras model
        model = Sequential()
        model.add(Dense(16, input_dim = z[~mask,].shape[1], activation = 'relu'))
        model.add(Dense(16, activation = 'relu'))
        model.add(Dense(1))
        
        # compile the keras model
        opt = keras.optimizers.RMSprop()
        mse = tf.keras.losses.MeanSquaredError()
        mae = tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)
        model.compile(loss=mse, optimizer= opt , metrics=mae)

        # Fit and predict dhat
        model.fit(z[~mask,], d[~mask,], epochs=num_epochs, batch_size=batch_size, verbose = 0)
        dhat = model.predict(z[mask,])
        
        # Fit and predict yhat
        model.fit(z[~mask,], y[~mask,], epochs=num_epochs, batch_size=batch_size, verbose = 0)
        yhat = model.predict(z[mask,])

        # Create array to save errors 
        dtil = np.zeros( len(z) ).reshape( len(z) , 1 )
        ytil = np.zeros( len(z) ).reshape( len(z) , 1 )

        # save errors  
        dtil[mask] =  d[mask,] - dhat.reshape( len(I[b]) , 1 )
        ytil[mask] = y[mask,] - yhat.reshape( len(I[b]) , 1 )
        print(b, " ")
    
    # Create dataframe 
    data_2 = pd.DataFrame(np.concatenate( ( ytil, dtil,clu ), axis = 1), columns = ['ytil','dtil','CountyCode'])
   
    # OLS clustering at the County level
    model = "ytil ~ dtil"
    rfit = smf.ols(model , data=data_2).fit().get_robustcov_results(cov_type = "cluster", groups= data_2['CountyCode'])
    
    coef_est = rfit.summary2().tables[1]['Coef.']['dtil']
    se = rfit.summary2().tables[1]['Std.Err.']['dtil']

    print("Coefficient is {}, SE is equal to {}".format(coef_est, se))
    
    return coef_est, se, dtil, ytil, rfit
    


# ## Estimating the effect with DML for neural nets

# In[13]:


# Create main variables
Y = rdata['logghomr']
D = rdata['logfssl']
Z = rdata.drop(['logghomr', 'logfssl', 'CountyCode'], axis=1)
CLU = rdata['CountyCode']

# as matrix
y = Y.to_numpy().reshape( len(Y) , 1 )
d = D.to_numpy().reshape( len(Y) , 1 )
z = Z.to_numpy()
clu = CLU.to_numpy().reshape( len(Y) , 1 )


# In[71]:


DML2_nn = DML2_for_NN(z, d, y, 2, clu, 100, 10)

