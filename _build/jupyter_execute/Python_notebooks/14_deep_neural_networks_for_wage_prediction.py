#!/usr/bin/env python
# coding: utf-8

# 
# 
# This notebook contains an example for teaching.
# 

# # Deep Neural Networks for Wage Prediction

# So far we considered many machine learning method, e.g Lasso and Random Forests, to build a predictive model. In this lab, we extend our toolbox by predicting wages by a neural network.

# ## Data preparation

# Again, we consider data from the U.S. March Supplement of the Current Population Survey (CPS) in 2015.

# 
# 
# This notebook contains an example for teaching.
# 
# 
# **Simple Case Study using Wage Data from 2015 - proceeding**
# 
# So far we considered many machine learning method, e.g Lasso and Random Forests, to build a predictive model. In this lab, we extend our toolbox by predicting wages by a neural network.
# 
# **Data preparation**
# 
# Again, we consider data from the U.S. March Supplement of the Current Population Survey (CPS) in 2015.

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


# In[2]:


rdata_read = pyreadr.read_r("../data/wage2015_subsample_inference.Rdata")
data = rdata_read[ 'data' ]
n = data.shape[0]


# In[3]:


# Import relevant packages for splitting data
import random
import math
# Set Seed
# to make the results replicable (generating random numbers)
np.random.seed(0)
random = np.random.randint(0, data.shape[0], size=math.floor(data.shape[0]))
data["random"] = random
random    # the array does not change 


# In[4]:


data_2 = data.sort_values(by=['random'])
data_2


# In[5]:


# Create training and testing sample 
data_train = data_2[ : math.floor(n*3/4)]    # training sample
data_test =  data_2[ math.floor(n*3/4) : ]   # testing sample
print(data_train.shape)
print(data_test.shape)


# In[6]:


data_train = data_train.iloc[:, 0:16]
data_test = data_test.iloc[:, 0:16] 
data_test


# In[7]:


# normalize the data
from sklearn.preprocessing import MinMaxScaler

scaler =  MinMaxScaler().fit(data_train)
scaler =  MinMaxScaler().fit(data_test)

# scaler = preprocessing.StandardScaler().fit(data_train)
# scaler = preprocessing.StandardScaler().fit(data_test)

data_train_scaled = scaler.transform(data_train)
data_test_scaled = scaler.transform(data_test)


# In[8]:


columns = list(data_train)


# In[9]:


data_train_scaled = pd.DataFrame(data_train_scaled, columns = columns)
data_test_scaled = pd.DataFrame(data_test_scaled, columns = columns)
data_test_scaled


# Then, we construct the inputs for our network.

# In[10]:


formula_basic = "lwage ~ sex + exp1 + shs + hsg+ scl + clg + mw + so + we"
Y_train, model_X_basic_train = patsy.dmatrices(formula_basic, data_train_scaled, return_type='dataframe')
Y_test, model_X_basic_test = patsy.dmatrices(formula_basic, data_test_scaled, return_type='dataframe')


# ## Neural Networks

# First, we need to determine the structure of our network. We are using the R/python package *keras* to build a simple sequential neural network with three dense layers.

# In[11]:


model_X_basic_train.shape[1]


# In[12]:


# define the keras model
model = Sequential()
model.add(Dense(20, input_dim = model_X_basic_train.shape[1], activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
# model.add(Dense(5, activation = 'relu'))

model.add(Dense(1))


# In[13]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[14]:


# compile the keras model
opt = keras.optimizers.Adam(learning_rate=0.005)
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)


# Let us have a look at the structure of our network in detail.

# In[15]:


model.compile(loss=mse, optimizer= opt , metrics=mae)
model.summary(line_length=None, positions=None, print_fn=None)


# It is worth to notice that we have in total $441$ trainable parameters.
# 
# Now, let us train the network. Note that this takes some computation time. Thus, we are using gpu to speed up. The exact speed-up varies based on a number of factors including model architecture, batch-size, input pipeline complexity, etc.

# In[16]:


# fit the keras model on the dataset
num_epochs = 1000


# Check this [link](https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network), to understand batch_size argument

# In[17]:


# fit the keras model on the dataset
model.fit(model_X_basic_train, Y_train, epochs=150, batch_size=10)


# In[18]:


model.metrics_names


# In[19]:


model.evaluate(model_X_basic_test, Y_test, verbose = 0)


# In[20]:


pred_nn = model.predict(model_X_basic_test)
pred_nn


# In[21]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[22]:


resid_basic = (Y_test-pred_nn)**2


# In[23]:


MSE_nn_basic = sm.OLS( resid_basic , np.ones( resid_basic.shape[0] ) ).fit().summary2().tables[1].iloc[0, 0:2]
MSE_nn_basic


# In[24]:


R2_nn_basic = 1 - ( MSE_nn_basic[0]/Y_test.var() )
print( f"The R^2 using NN is equal to = {R2_nn_basic[0]}" ) # MSE NN (basic model) 

