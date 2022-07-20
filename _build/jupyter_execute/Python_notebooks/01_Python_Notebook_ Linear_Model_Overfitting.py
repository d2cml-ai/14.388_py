#!/usr/bin/env python
# coding: utf-8

# * Python code replication of:
# " https://www.kaggle.com/victorchernozhukov/r-notebook-linear-model-overfiting "
# * Created by: Alexander Quispe

# # Simple Exercise on Overfitting
# ## 1. First set p=n
# 

# In[1]:


import numpy as np
import random
import statsmodels.api as sm


# In[2]:


# Set Seed
random.seed(10)
print(random.random())

n = 1000
p = n


# In[3]:


X = np.random.normal(0, 1, size=(n, p))
Y = np.random.normal(0, 1,n)


# In[4]:


mod = sm.OLS(Y, X)    # Describe model
res = mod.fit()
print(res.summary())


# In[5]:


print("p/n is")
print(p/n)


# In[6]:


print("R2 is")
res.rsquared


# In[7]:


print("Adjusted R2 is")
est2 = mod.fit()
est2.rsquared_adj


# ## 2. Second, set p=n/2.

# In[8]:


random.seed(10)
n = 1000
p = n/2


# In[9]:


X = np.random.normal(0, 1, size=(n, int(p)))
Y = np.random.normal(0, 1,n)
mod = sm.OLS(Y, X)    # Describe model
res = mod.fit()
print(res.summary())


# In[10]:


print("p/n is")
print(p/n)

print("R2 is")
res.rsquared

print("Adjusted R2 is")
est2 = mod.fit()
est2.rsquared_adj


# In[11]:


print("p/n is \n",p/n )
#print("summary()\n",res.summary())
print("rsquared\n",est2.rsquared)
print("rsquared_adj\n",est2.rsquared_adj)


# ## 3. Third, set p/n =.05

# In[12]:


random.seed(10)
n = 1000
p = 0.05*n
int(p)


# In[13]:


X = np.random.normal(0, 1, size=(n, int(p)))
Y = np.random.normal(0, 1,n)
mod = sm.OLS(Y, X)    # Describe model
res = mod.fit()
print(res.summary())


# In[14]:


print("p/n is \n",p/n )
#print("summary()\n",res.summary())
print("rsquared\n",res.rsquared)
print("rsquared_adj\n",res.rsquared_adj)

