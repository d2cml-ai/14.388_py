#!/usr/bin/env python
# coding: utf-8

# # A Simple Example of Properties of IV estimator when Instruments are Weak

# Simulation Design

# In[1]:


import hdmpy
import numpy as np
import random
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from statsmodels.sandbox.regression.gmm import IV2SLS
import numpy as np                                                              
import seaborn as sns                                                           
from scipy import stats                                                         
import matplotlib.pyplot as plt 


# In[2]:


# Simulation Design

# Set seed
np.random.seed(0)
B = 1000
IVEst = np.zeros( B )
n = 100
beta = .25

mean = 0
sd = 1

U = np.random.normal( mean , sd, n ).reshape( n, 1 )
Z = np.random.normal( mean , sd, n ).reshape( n, 1 )
D = beta*Z + U 
Y = D + U

mod = sm.OLS(D, sm.add_constant(Z))    # Describe model
res = mod.fit()
print(res.summary())


# In[3]:


IV = IV2SLS(Y, D, sm.add_constant(Z))
IV_res = IV.fit()
print(IV_res.summary())


# In[4]:


IV_res.summary2().tables[1]["Coef."][0]


# Note that the instrument is weak here (contolled by $\beta$) -- the t-stat is less than 4.

# # Run 1000 trials to evaluate distribution of the IV estimator

# In[5]:


# Simulation design 

# Set seed
np.random.seed(0)
B = 1000 # Trials
IVEst = np.zeros( B )

for i in range( 0, B ):
    U = np.random.normal( mean , sd, n ).reshape( n, 1 )
    Z = np.random.normal( mean , sd, n ).reshape( n, 1 )
    D = beta*Z + U 
    Y = D + U
    
    IV = IV2SLS(Y, D, sm.add_constant(Z))
    IV_res = IV.fit()
    
    IVEst[ i ] = IV_res.summary2().tables[1]["Coef."][0]


# In[6]:


IVEst


# # Plot the Actual Distribution against the Normal Approximation (based on Strong Instrument Assumption)

# In[7]:


val = np.arange(-5,5.5,0.05)
var = (1/beta**2)*(1/100)   # theoretical variance of IV
sd = np.sqrt(var)

normal_dist = np.random.normal(0,sd,val.shape[0])

# plotting both distibutions on the same figure
fig = sns.kdeplot(IVEst-1, shade=True, color="r")
fig = sns.kdeplot(normal_dist, shade=True, color="b")

plt.title("Actual Distribution vs Gaussian")
plt.xlabel('IV Estimator -True Effect')
plt.xlim(-5,5)


# In[8]:


rejection_frequency = np.sum(( np.abs(IVEst-1)/sd > 1.96))/B
print("Rejection Frequency is {} ,while we expect it to be .05".format(rejection_frequency))


# In[ ]:




