#!/usr/bin/env python
# coding: utf-8

# # AutoML for wage prediction

# ## Automatic Machine Learning with H2O AutoML using Wage Data from 2015

# We illustrate how to predict an outcome variable Y in a high-dimensional setting, using the AutoML package *H2O* that covers the complete pipeline from the raw dataset to the deployable machine learning model. In last few years, AutoML or automated machine learning has become widely popular among data science community. 

# We can use AutoML as a benchmark and compare it to the methods that we used in the previous notebook where we applied one machine learning method after the other.

# In[1]:


# Import relevant packages
import pandas as pd
import numpy as np
import pyreadr
import os
from urllib.request import urlopen
from sklearn import preprocessing
import patsy
from h2o.automl import H2OAutoML

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#pip install h2o


# In[3]:


# load the H2O package
import h2o

# start h2o cluster
h2o.init()


# In[4]:


link="https://raw.githubusercontent.com/d2cml-ai/14.388_py/main/data/wage2015_subsample_inference.Rdata"
response = urlopen(link)
content = response.read()
fhandle = open( 'wage2015_subsample_inference.Rdata', 'wb')
fhandle.write(content)
fhandle.close()
result = pyreadr.read_r("wage2015_subsample_inference.Rdata")
os.remove("wage2015_subsample_inference.Rdata")

# Extracting the data frame from rdata_read
data = result[ 'data' ]
n = data.shape[0]
type(data)


# In[5]:


# Import relevant packages for splitting data
import random
import math

# Set Seed
# to make the results replicable (generating random numbers)
np.random.seed(0)
random = np.random.randint(0, data.shape[0], size=math.floor(data.shape[0]))
data["random"] = random
random    # the array does not change 
data_2 = data.sort_values(by=['random'])


# In[6]:


# Create training and testing sample 
train = data_2[ : math.floor(n*3/4)]    # training sample
test =  data_2[ math.floor(n*3/4) : ]   # testing sample
print(train.shape)
print(test.shape)


# In[7]:


# start h2o cluster
h2o.init()


# In[8]:


# convert data as h2o type
train_h = h2o.H2OFrame(train)
test_h = h2o.H2OFrame(test)

# have a look at the data
train_h.describe()


# In[9]:


# define the variables
y = 'lwage'

data_columns = list(data)
no_relev_col = ['wage','occ2', 'ind2', 'random', 'lwage']

# This gives us: new_list = ['carrot' , 'lemon']
x = [col for col in data_columns if col not in no_relev_col]


# In[10]:


# run AutoML for 10 base models and a maximal runtime of 100 seconds
# Run AutoML for 30 seconds
aml = H2OAutoML(max_runtime_secs = 100, max_models = 10, seed = 1)
aml.train(x = x, y = y, training_frame = train_h, leaderboard_frame = test_h)


# In[11]:


# AutoML Leaderboard
lb = aml.leaderboard
print(lb)


# We see that two Stacked Ensembles are at the top of the leaderboard. Stacked Ensembles often outperform a single model. The out-of-sample (test) MSE of the leading model is given by

# In[12]:


aml.leaderboard['mse'][0,0]


# The in-sample performance can be evaluated by

# In[13]:


aml.leader


# This is in line with our previous results. To understand how the ensemble works, let's take a peek inside the Stacked Ensemble "All Models" model.  The "All Models" ensemble is an ensemble of all of the individual models in the AutoML run.  This is often the top performing model on the leaderboard.

# In[14]:


model_ids = h2o.as_list(aml.leaderboard['model_id'][0], use_pandas=True)


# In[15]:


model = model_ids[model_ids['model_id'].str.contains("StackedEnsemble_AllModels")].values.tolist()
model_id = model[0][0]
model_id


# In[16]:


se = h2o.get_model(model_id)
se


# In[17]:


# Get the Stacked Ensemble metalearner model
metalearner = se.metalearner()
metalearner


# Examine the variable importance of the metalearner (combiner) algorithm in the ensemble. This shows us how much each base learner is contributing to the ensemble. The AutoML Stacked Ensembles use the default metalearner algorithm (GLM with non-negative weights), so the variable importance of the metalearner is actually the standardized coefficient magnitudes of the GLM.

# The table above gives us the variable importance of the metalearner in the ensemble. The AutoML Stacked Ensembles use the default metalearner algorithm (GLM with non-negative weights), so the variable importance of the metalearner is actually the standardized coefficient magnitudes of the GLM. 
# 

# In[18]:


metalearner.coef_norm()


# In[19]:


metalearner.std_coef_plot()


# In[20]:


h2o.get_model(model_id).metalearner()


# ## Generating Predictions Using Leader Model
# 
# We can also generate predictions on a test sample using the leader model object.

# In[21]:


pred = aml.predict(test_h)
pred.head()


# This allows us to estimate the out-of-sample (test) MSE and the standard error as well.

# In[22]:


pred_2 = pred.as_data_frame()
pred_aml = pred_2.to_numpy()


# In[23]:


Y_test = test_h['lwage'].as_data_frame().to_numpy()


# In[24]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[25]:


resid_basic = (Y_test-pred_aml)**2

MSE_aml_basic = sm.OLS( resid_basic , np.ones( resid_basic.shape[0] ) ).fit().summary2().tables[1].iloc[0, 0:2]
MSE_aml_basic


# We observe both a lower MSE and a lower standard error compared to our previous results (see [here](https://www.kaggle.com/janniskueck/pm3-notebook-newdata)).

# ### By using model_performance()
# If needed, the standard model_performance() method can be applied to the AutoML leader model and a test set to generate an H2O model performance object.
# 
# 

# In[26]:


perf = aml.leader.model_performance(test_h)
perf

