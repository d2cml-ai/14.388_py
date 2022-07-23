#!/usr/bin/env python
# coding: utf-8

# # Sensitivity Analysis with Sensmakr and Debiased ML

# ## Sensititivy Analysis for Unobserved Confounder with DML and Sensmakr
# 
# 
#  Here we experiment with using package "sensemakr" in conjunction with debiased ML

# We will 
# 
# * mimic the partialling out procedure with machine learning tools, 
# 
# * and invoke Sensmakr to compute $\phi^2$ and plot sensitivity results.
# 

# We will use the sensemakr package adapted in python (PySensemakr) by Brian Hill and Nathan LaPierre [ink](https://github.com/nlapier2/PySensemakr)

# In[1]:


#pip install PySensemakr


# In[2]:


#Import packages
import sensemakr as smkr
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# loads data
darfur = pd.read_csv("../data/darfur.csv")
darfur.shape


# Data is described here 
# https://cran.r-project.org/web/packages/sensemakr/vignettes/sensemakr.html
# 
# The main outcome is attitude towards peace -- the peacefactor.
# The key variable of interest is whether the responders were directly harmed (directlyharmed).
# We want to know if being directly harmed in the conflict causes people to support peace-enforcing measures.
# The measured confounders include female indicator, age, farmer, herder, voted in the past, and household size.
# There is also a village indicator, which we will treat as fixed effect and partial it out before conducting
# the analysis. The standard errors will be clustered at the village level.

# ## Take out village fixed effects and run basic linear analysis

# In[4]:


# get rid of village fixed effects
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[5]:


# 1. basic model
peacefactorR = smf.ols('peacefactor~village' , data=darfur).fit().resid
directlyharmedR = smf.ols('directlyharmed~village' , data=darfur).fit().resid
femaleR = smf.ols('female~village' , data=darfur).fit().resid
ageR = smf.ols('age~village' , data=darfur).fit().resid
farmerR = smf.ols('farmer_dar~village' , data=darfur).fit().resid
herderR = smf.ols('herder_dar~village' , data=darfur).fit().resid
pastvotedR = smf.ols('pastvoted~village' , data=darfur).fit().resid
hhsizeR = smf.ols('hhsize_darfur~village' , data=darfur).fit().resid


# In[6]:


darfurR = pd.concat([peacefactorR, directlyharmedR, femaleR,
                    ageR, farmerR, herderR, pastvotedR, 
                     hhsizeR, darfur['village']], axis=1)
darfurR.head()


# In[7]:


darfurR.columns = ["peacefactorR", "directlyharmedR", "femaleR",
                    "ageR", "farmerR", "herderR", "pastvotedR", 
                     "hhsize_darfurR", "village"]
darfurR.head()


# In[8]:


# Preliminary linear model analysis 
# Linear model 1 
linear_model_1 = smf.ols('peacefactorR~ directlyharmedR+ femaleR + ageR + farmerR+ herderR + pastvotedR + hhsizeR' 
        ,data=darfurR ).fit().get_robustcov_results(cov_type = "cluster", groups= darfurR['village'])
linear_model_1_table = linear_model_1.summary2().tables[1]
linear_model_1_table


# In[9]:


# Linear model 2 
linear_model_2 = smf.ols('peacefactorR~ femaleR + ageR + farmerR+ herderR + pastvotedR + hhsizeR' 
        ,data=darfurR ).fit().get_robustcov_results(cov_type = "cluster", groups= darfurR['village'])
linear_model_2_table = linear_model_2.summary2().tables[1]
linear_model_2_table


# In[10]:


# Linear model 3
linear_model_3 = smf.ols('directlyharmedR~ femaleR + ageR + farmerR+ herderR + pastvotedR + hhsizeR' 
        ,data=darfurR ).fit().get_robustcov_results(cov_type = "cluster", groups= darfurR['village'])
linear_model_3_table = linear_model_3.summary2().tables[1]
linear_model_3_table


# ## We first use Lasso for Partilling Out Controls

# In[11]:


import hdmpy
import patsy 
from patsy import ModelDesc, Term, EvalFactor


# In[12]:



X = patsy.dmatrix("(femaleR + ageR + farmerR+ herderR + pastvotedR + hhsizeR)**3", darfurR)
Y = darfurR['peacefactorR'].to_numpy()
D = darfurR['directlyharmedR'].to_numpy()


# In[13]:


resY = hdmpy.rlasso(X,Y, post = False).est['residuals'].reshape( Y.size,)
resD = hdmpy.rlasso(X,D, post = False).est['residuals'].reshape( D.size,)


# In[14]:


FVU_Y = 1 - np.var(resY)/np.var(peacefactorR)
FVU_D = 1 - np.var(resD)/np.var(directlyharmedR)

print("Controls explain the following fraction of variance of Outcome", FVU_Y)
print("Controls explain the following fraction of variance of treatment", FVU_D)


# In[15]:


darfurR['resY'] = resY
darfurR['resD'] = resD


# In[16]:


# Filan estimation
# Culster SE by village
dml_darfur_model = smf.ols('resY~ resD',data=darfurR ).fit().get_robustcov_results(cov_type = "cluster", groups= darfurR['village'])
dml_darfur_model_table = dml_darfur_model.summary2().tables[1]
dml_darfur_model_table


# ## Manual Bias Analysis

# In[17]:


# linear model to use as input in sensemakr   
dml_darfur_model= smf.ols('resY~ resD',data=darfurR ).fit()
dml_darfur_model_table = dml_darfur_model.summary2().tables[1]
dml_darfur_model_table


# In[18]:


beta = dml_darfur_model_table['Coef.'][1]
beta


# In[19]:


# Hypothetical values of partial R2s 
R2_YC = .16 
R2_DC = .01

# Elements of the formal
kappa = (R2_YC * R2_DC)/(1- R2_DC)
varianceRatio = np.mean(dml_darfur_model.resid**2)/np.mean(dml_darfur_model.resid**2)

# Compute square bias 
BiasSq =  kappa*varianceRatio

# Compute absolute value of the bias
print(np.sqrt(BiasSq))

# plotting 
gridR2_DC = np.arange(0,0.3,0.001)
gridR2_YC =  kappa*(1 - gridR2_DC)/gridR2_DC
gridR2_YC = np.where(gridR2_YC > 1, 1, gridR2_YC)

plt.title("Combo of R2 such that |Bias|<{}".format(round(np.sqrt(BiasSq), 5)))
plt.xlabel("Partial R2 of Treatment with Confounder") 
plt.ylabel("Partial R2 of Outcome with Confounder") 
plt.plot(gridR2_DC,gridR2_YC) 
plt.show()


# ## Bias Analysis with Sensemakr

# In[20]:


# Imports
import sensemakr as smkr
# from sensemakr import sensitivity_stats
# from sensemakr import bias_functions
# from sensemakr import ovb_bounds
# from sensemakr import ovb_plots
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


# In[21]:


a = 1
b = 3


# In[22]:


if a is not None and b is not None:
    print('hola')


# In[23]:


import sensemakr as smkr


# In[24]:


# We need to double check why the function does not allow to run withour the benchmark_covariates argument
model = smkr.Sensemakr( model = dml_darfur_model, treatment = "resD")


# In[25]:


model.summary()


# In[26]:


model.plot()


# ## Next We use Random Forest as ML tool for Partialling Out

# The following code does DML with clsutered standard errors by ClusterID

# In[27]:


import itertools
from itertools import compress


# In[28]:


def DML2_for_PLM(x, d, y, dreg, yreg, nfold, clu):
    
    # Num ob observations
    nobs = x.shape[0]
    
    # Define folds indices 
    list_1 = [*range(0, nfold, 1)]*nobs
    sample = np.random.choice(nobs,nobs, replace=False).tolist()
    foldid = [list_1[index] for index in sample]

    # Create split function(similar to R)
    def split(z, f):
        count = max(f) + 1
        return tuple( list(itertools.compress(z, (el == i for el in f))) for i in range(count) ) 

    # Split observation indices into folds 
    list_2 = [*range(0, nobs, 1)]
    I = split(list_2, foldid)
    
    # loop to save results
    for b in range(0,len(I)):
    
        # Split data - index to keep are in mask as booleans
        include_idx = set(I[b])  #Here should go I[b] Set is more efficient, but doesn't reorder your elements if that is desireable
        mask = np.array([(i in include_idx) for i in range(len(x))])

        # Lasso regression, excluding folds selected 
        dfit = dreg(x[~mask,], d[~mask,])
        yfit = yreg(x[~mask,], y[~mask,])

        # predict estimates using the 
        dhat = dfit.predict( x[mask,] )
        yhat = yfit.predict( x[mask,] )

        # Create array to save errors 
        dtil = np.zeros( len(x) ).reshape( len(x) , 1 )
        ytil = np.zeros( len(x) ).reshape( len(x) , 1 )

        # save errors  
        dtil[mask] =  d[mask,] - dhat.reshape( len(I[b]) , 1 )
        ytil[mask] = y[mask,] - yhat.reshape( len(I[b]) , 1 )
        print(b, " ")
    
    # Create dataframe 
    data_2 = pd.DataFrame(np.concatenate( ( ytil, dtil,clu ), axis = 1), columns = ['ytil','dtil','CountyCode'])
   
     # OLS clustering at the County level
    model = "ytil ~ dtil"
    baseline_ols = smf.ols(model , data=data_2).fit().get_robustcov_results(cov_type = "cluster", groups= data_2['CountyCode'])
    coef_est = baseline_ols.summary2().tables[1]['Coef.']['dtil']
    se = baseline_ols.summary2().tables[1]['Std.Err.']['dtil']

    print("Coefficient is {}, SE is equal to {}".format(coef_est, se))
    
    return coef_est, se, dtil, ytil, data_2
    #return dtil, ytil, data_2

    


# In[29]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder


# In[30]:


# This new matrix include intercept
x = patsy.dmatrix("~  femaleR + ageR + farmerR + herderR + pastvotedR + hhsizeR", darfurR)
y = darfurR['peacefactorR'].to_numpy().reshape( len(Y) , 1 )
d = darfurR['directlyharmedR'].to_numpy().reshape( len(Y) , 1 )


# In[31]:


darfurR['village'].unique().size


# In[32]:


# creating instance of labelencoder
labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column
darfurR['village_clu'] = labelencoder.fit_transform(darfurR['village'])

# Create cluster object
CLU = darfurR['village_clu']
clu = CLU.to_numpy().reshape( len(Y) , 1 )


# In[33]:


#DML with RF
def dreg(x,d):
    result = RandomForestRegressor( random_state = 0, 
                                   n_estimators = 500, 
                                   max_features = max( int( x.shape[1] / 3 ), 1 ), 
                                   min_samples_leaf = 1 ).fit( x, d )
    return result

def yreg(x,y):
    result = RandomForestRegressor( random_state = 0, 
                                   n_estimators = 500, 
                                   max_features = max( int( x.shape[1] / 3 ), 1 ), 
                                   min_samples_leaf = 1 ).fit( x, y )
    return result

DML2_RF = DML2_for_PLM(x, d, y, dreg, yreg, 10, clu)   # set to 2 due to computation time


# In[44]:


resY = DML2_RF[2]
resD = DML2_RF[3]

FVU_Y = max(1 - ( np.var(resY)/np.var(peacefactorR) ), 0 )
FVU_D = max(1 - ( np.var(resD)/np.var(directlyharmedR) ), 0 )

print("Controls explain the following fraction of variance of Outcome", FVU_Y)
print("Controls explain the following fraction of variance of treatment", FVU_D)


# In[45]:


darfurR['resY_rf'] = resY
darfurR['resD_rf'] = resD

# linear model to use as input in sensemakr   
dml_darfur_model_rf= smf.ols('resY_rf~ resD_rf',data=darfurR ).fit()
dml_darfur_model_rf_table = dml_darfur_model_rf.summary2().tables[1]


# In[46]:


# We need to double check why the function does not allow to run withour the benchmark_covariates argument
dml_darfur_sensitivity = smkr.Sensemakr(dml_darfur_model_rf, "resD_rf", benchmark_covariates = "Intercept")
dml_darfur_sensitivity.summary()

# Make a contour plot for the estimate
dml_darfur_sensitivity.plot()

