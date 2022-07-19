#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[1]:


# Imports
from sensemakr import sensemakr
from sensemakr import sensitivity_stats
from sensemakr import bias_functions
from sensemakr import ovb_bounds
from sensemakr import ovb_plots
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')


# In[3]:


get_ipython().run_line_magic('autoreload', '2')


# In[4]:


# loads data
darfur = pd.read_csv("../data/darfur.csv")
darfur.head()


# In[5]:


# runs regression model
reg_model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + '                    'pastvoted + hhsize_darfur + female + village', data=darfur)
model = reg_model.fit()


# In[6]:


# Define parameters for sensemakr
treatment = "directlyharmed"
q = 1.0
alpha = 0.25
reduce = True
benchmark_covariates=["female"]
kd = [1, 2, 3]
ky = kd


# In[18]:


# Create a sensemakr object and print summary of results
s = sensemakr.Sensemakr(model, treatment, q=q, alpha=alpha, reduce=reduce, benchmark_covariates=benchmark_covariates, kd=kd)
s.summary()


# In[19]:


# Make a contour plot for the estimate
ovb_plots.ovb_contour_plot(sense_obj=s, sensitivity_of='estimate')


# In[ ]:




