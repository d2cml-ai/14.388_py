#!/usr/bin/env python
# coding: utf-8

# This notebook contains an example for teaching.
# 

# # Automatic Machine Learning with H2O AutoML using Wage Data from 2015

# We illustrate how to predict an outcome variable Y in a high-dimensional setting, using the AutoML package *H2O* that covers the complete pipeline from the raw dataset to the deployable machine learning model. In last few years, AutoML or automated machine learning has become widely popular among data science community. 

# We can use AutoML as a benchmark and compare it to the methods that we used in the previous notebook where we applied one machine learning method after the other.

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


# load the H2O package
import h2o

# start h2o cluster
h2o.init()


# In[ ]:


# load dataset
rdata_read = pyreadr.read_r("../data/wage2015_subsample_inference.Rdata")
data = rdata_read[ 'data' ]
n = data.shape[0]

type(data)


# In[ ]:


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


# In[ ]:


# Create training and testing sample 
train = data_2[ : math.floor(n*3/4)]    # training sample
test =  data_2[ math.floor(n*3/4) : ]   # testing sample
print(data_train.shape)
print(data_test.shape)


# In[ ]:


# start h2o cluster
h2o.init()


# In[ ]:


# convert data as h2o type
train_h = h2o.H2OFrame(train)
test_h = h2o.H2OFrame(test)

# have a look at the data
train_h.describe()


# In[ ]:


# define the variables
y = 'lwage'

data_columns = list(data)
no_relev_col = ['wage','occ2', 'ind2', 'random', 'lwage']

# This gives us: new_list = ['carrot' , 'lemon']
x = [col for col in data_columns if col not in no_relev_col]


# In[ ]:


# run AutoML for 10 base models and a maximal runtime of 100 seconds
# Run AutoML for 30 seconds
aml = H2OAutoML(max_runtime_secs = 100, max_models = 10, seed = 1)
aml.train(x = x, y = y, training_frame = train_h, leaderboard_frame = test_h)


# In[ ]:


# AutoML Leaderboard
lb = aml.leaderboard
print(lb)


# We see that two Stacked Ensembles are at the top of the leaderboard. Stacked Ensembles often outperform a single model. The out-of-sample (test) MSE of the leading model is given by

# In[ ]:


aml.leaderboard['mse'][0,0]


# The in-sample performance can be evaluated by

# In[ ]:


aml.leader


# This is in line with our previous results. To understand how the ensemble works, let's take a peek inside the Stacked Ensemble "All Models" model.  The "All Models" ensemble is an ensemble of all of the individual models in the AutoML run.  This is often the top performing model on the leaderboard.

# In[ ]:


model_ids = h2o.as_list(aml.leaderboard['model_id'][0], use_pandas=True)


# In[ ]:


model = model_ids[model_ids['model_id'].str.contains("StackedEnsemble_AllModels")].values.tolist()
model_id = model[0][0]
model_id


# In[ ]:


se = h2o.get_model('StackedEnsemble_AllModels_AutoML_20210420_101446')
se


# In[ ]:


# Get the Stacked Ensemble metalearner model
metalearner = se.metalearner()
metalearner


# Examine the variable importance of the metalearner (combiner) algorithm in the ensemble. This shows us how much each base learner is contributing to the ensemble. The AutoML Stacked Ensembles use the default metalearner algorithm (GLM with non-negative weights), so the variable importance of the metalearner is actually the standardized coefficient magnitudes of the GLM.

# The table above gives us the variable importance of the metalearner in the ensemble. The AutoML Stacked Ensembles use the default metalearner algorithm (GLM with non-negative weights), so the variable importance of the metalearner is actually the standardized coefficient magnitudes of the GLM. 
# 

# In[ ]:


metalearner.coef_norm()


# In[ ]:


metalearner.std_coef_plot()


# In[ ]:


h2o.get_model(model_id).metalearner()


# ## Generating Predictions Using Leader Model
# 
# We can also generate predictions on a test sample using the leader model object.

# In[ ]:


pred = aml.predict(test_h)
pred.head()


# This allows us to estimate the out-of-sample (test) MSE and the standard error as well.

# In[ ]:


pred_2 = pred.as_data_frame()
pred_aml = pred_2.to_numpy()


# In[ ]:


Y_test = test_h['lwage'].as_data_frame().to_numpy()


# In[ ]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[ ]:


resid_basic = (Y_test-pred_aml)**2

MSE_aml_basic = sm.OLS( resid_basic , np.ones( resid_basic.shape[0] ) ).fit().summary2().tables[1].iloc[0, 0:2]
MSE_aml_basic


# We observe both a lower MSE and a lower standard error compared to our previous results (see [here](https://www.kaggle.com/janniskueck/pm3-notebook-newdata)).

# ### By using model_performance()
# If needed, the standard model_performance() method can be applied to the AutoML leader model and a test set to generate an H2O model performance object.
# 
# 

# In[ ]:


perf = aml.leader.model_performance(test_h)
perf

