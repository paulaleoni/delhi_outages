#!/usr/bin/env python
# coding: utf-8

# # bunching estimation using the real data
# 
# take concept from simulation

# In[2]:


# load packages

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodels.api as sm
import numpy as np
#from scipy.optimize import fsolve
#from scipy import integrate
from sympy import symbols, solve, Eq
import random
from heapq import nlargest
import tools # this imports the file tools.py // must be in the same folder as the current file
import zipfile

np.random.seed(111)


# In[73]:


# load data
wd = Path.cwd()
folder = 'stata'
file = 'Outages_{period}.dta'
zip = 'Outages_{period}.zip'
zip = zipfile.ZipFile(wd.parent/folder/zip)
data = pd.read_stata(zip.open(file)) # convert_categoricals = True
#print(data.columns)


data['year'] = data.apply(lambda row: row.date.year, axis=1)

# keep year 2019
#data = data[data.apply(lambda row: row.date.year == 2019, axis=1)]

# keep only unplanned
data = data.loc[data.planned != 'planned',]

# at least 100 custumers affected
data = data.loc[data.noofcustomersaffected >= 100]

data.reset_index(drop = True)

# there is probably some rounding issue in unserveredmuduetooutage
# if it is 0 we replace it with 0.00001
data.loc[data.unservedmuduetooutage.isna() ,'unservedmuduetooutage'] = 0.00001


# keep one firm
#data = data.loc[data.discom == 'tata',]
#data.describe()


# In[25]:


bunching = {}
for f in data.discom.unique():
    bunching[f] = {'data': data[data.discom == f]}

bunching['pooled'] = {'data': data} 


# density of duration 

# In[29]:


xmax  = 420

fig, axs = plt.subplots(1,2,figsize=(18,5))
for f in bunching:
    dt = bunching[f]['data']
    sns.kdeplot(dt.duration_minutes, ax = axs[0], label = f)
    axs[1].hist(dt.duration_minutes, histtype = 'step', bins = xmax, label = f)

axs[0].axvline(60, color  = 'grey')
axs[0].axvline(120, color  = 'grey')
axs[0].set_xlim(0,xmax)
axs[0].legend()

axs[1].axvline(60, color  = 'grey')
axs[1].axvline(120, color  = 'grey')
axs[1].set_xlim(0,xmax)
axs[1].legend()


# # Bunching Estimation

# - separately for each firm and notch
# - define, binsize, excluded region, missing region, maximum  and minimum value of x considered and the degree of the poynomial (6 in the moment)
# - preparation steps
#     - create a dataframe at the bin-level by sorting all observations (duration) of the dataframe in bins with binsize as given
#     - count the observations being in each bin
#     - create polynomials of the lowerbound of each bin up the degree as given
#     - create dummy variables for bunching and missing mass
# - estimation
#     - create a regressor matrix using an intercept, the polynomials and the dummy variables
#     - OLS of #observations on the regressor matrix
# - prediction
#     - set all dummies to zero
#     - use the estimated coefficients to predict the number of observations in each bin
# - $\Delta x$ (total_bunch? - maybe be should switch our wording to avoid confusion)
#     - extract the prediction and the actual observations in the excluded region 
#     - using the estimated model parameters, approximate a function
#     - integrate this function from the notch until notch + z 
#     - get the difference between observation and prediction
#     - from that difference, substract the calculated integral and find the value of z for which it is zero
#     - the found z is our $\Delta x$

# In[31]:


# bunching at x = 60
bsize = 1
ex_reg = 10
ex_reg_miss = 20
z = 59
z_lower60 = z - bsize*ex_reg

missing60 = z + ex_reg_miss*bsize

for f in bunching:
    dt = bunching[f]['data']
    bunch60 = tools.bunching(dt.duration_minutes, bsize = bsize, xmax= 115, xmin= 0, z_upper= z, z_lower= z_lower60, missing = missing60, ex_reg= ex_reg, ex_reg_miss=ex_reg_miss, poly_dgr=6)
    bunching[f].update({'bunch60': bunch60})    
    #display(bunch60.estimation_res())
    print(f'-----bunching at 60 -----{f}----------')
    print('EX:', bunch60.get_deltaX(), 'mX:', bunch60.get_mX(), 'B:', bunch60.get_B())
    print('total bunching',bunch60.total_bunch())


#print(bunch60.df_count())



# In[33]:


# bunching at x = 120
bsize = 1
ex_reg = 10
z = 119
z_lower120 = z - 5 * bsize
missing120 = z + ex_reg * bsize


for f in bunching:
    dt = bunching[f]['data']
    bunch120 = tools.bunching(dt.duration_minutes, bsize = bsize, xmax= 160, xmin= 90, z_upper= z, z_lower= z_lower120, missing = missing120, ex_reg= ex_reg, ex_reg_miss = ex_reg_miss, poly_dgr=9, include_missing=False)
    bunching[f].update({'bunch120': bunch120})
    #display(bunch120.estimation_res())
    print(f'-----bunching at 120 -----{f}----------')
    print('deltaX:', bunch120.get_deltaX())
    print('total bunching',bunch120.total_bunch())

#print(bunch120.df_count())


# plot of counterfactual and actual data

# In[48]:



fig, ax = plt.subplots(1,len(bunching),figsize=(15,5))

for i in range(len(bunching)):
    f = list(bunching.keys())[i]
    # concat the two predictions
    pred60 = bunching[f]['bunch60'].prediction()
    pred120 = bunching[f]['bunch120'].prediction()
    pred = pd.concat([pred60,pred120], ignore_index=True)

    ax[i].plot(pred.duration, pred.y_pred, label = 'prediction')
    ax[i].plot(pred.duration, pred.nobs, label = 'observation')
    ax[i].axvline(60, color= 'grey', linestyle = 'dashed')
    ax[i].axvline(120, color= 'grey', linestyle = 'dashed')
    ax[i].set_xticks(ticks = [60, 120])
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].set_xlabel('Duration (Minutes)')
    ax[i].set_title(f)
    ax[i].legend()


# Visualize $\Delta x$

# In[49]:


fig, ax = plt.subplots(1,len(bunching),figsize=(15,5))
min = -50
for i in range(len(bunching)):
    f = list(bunching.keys())[i]
    bunch = bunching[f]['bunch60']
    pred60 = bunching[f]['bunch60'].prediction()
    b = np.round(60 + bunch.total_bunch())
    ax[i].plot(pred60.duration[pred60.duration >= 0], pred60.y_pred[pred60.duration >= 0], color = 'black', linestyle='dashed', label = 'counterfactual')
    ax[i].plot(pred60.duration, pred60.nobs, color = 'violet')
    ax[i].plot([60,60], [min,pred60.loc[pred60.duration == 60, 'y_pred'].reset_index(drop=True)[0]], color = 'grey')
    ax[i].axvline(b,color='grey', linestyle = 'dashed')
    ax[i].set_xlabel('Duration (Minutes)')
    ax[i].set_ylabel('# observations')
    ax[i].set_xticks([60 , b])
    ax[i].set_xticklabels(labels = [60,b])
    ax[i].set_ylim(min,pred60.nobs.max()+ 100)
    ax[i].fill_between(np.linspace(60,b,np.sum((pred60.duration > 60) & (pred60.duration < b))),pred60.y_pred[(pred60.duration > 60) & (pred60.duration < b)], np.repeat(min,np.sum((pred60.duration > 60) & (pred60.duration < b))), alpha = .3, color = 'grey')
    ax[i].annotate('bunching mass', xy = (67,min + 50), size = 7)
    ax[i].legend()
    ax[i].set_title(f)

#fig.savefig('xdensity.png',dpi=120, format='png')


# ## get counterfactual duration at outage level
# 
# !! this is maybe not optimal. Right now, I am assuming a poisson distribution in the bins of the missing mass. \
# 

# - separately for each firm
# - first, create a dataframe using the predictions from bunching estimation on bin-level
#     - create dummies for bins being in the bunching and missing mass at 60 and 120
#     - calculatie difference between number of observation in each bin and the prediction
#     - calculate probability of an observation being in a specific bin: prediction/sum(prediction)
# - second
#     - for each bin in the bunching region
#     - take difference in each bin, identify the observations (original data) that fall in that bin and randomly sample from those with size = difference
#     - randomly select a bin from the missing mass weighted by the probability
#     - redistribute observations from the selected bunching-bin to the selected missing-bin using a poisson distribution
# - save this as column 'duration_cf' in the dataframe
# 

# In[65]:


for f in bunching:
    bunch60 = bunching[f]['bunch60']
    bunch120 = bunching[f]['bunch120']
    # concat the two predictions
    pred60 = bunch60.prediction()
    pred120 = bunch120.prediction()
    pred60 = pred60[pred60.duration <= 100]
    pred120 = pred120[pred120.duration > 100]
    cf = pd.concat([pred60,pred120], ignore_index=True)
    # first need to get the counterfactuals for the bins    
    cf = cf.loc[:,['bin', 'nobs', 'duration', 'b', 'm','y_pred']]
    cf['difference'] = cf.nobs - cf.y_pred
    # make sure that at least one bin is in missing
    upper60 = 60 + abs(bunch60.total_bunch()) + bsize 
    upper120 = 120 + abs(bunch120.total_bunch()) + bsize 
    # create dummies
    cf['b60'] = cf.apply(lambda row: 1 if (row.b ==1) & (row.duration <= missing60) else 0, axis = 1)
    cf['missing60'] = cf.apply(lambda row: 1 if (row.duration > 60) & (row.duration <= upper60) else 0, axis = 1)
    cf['b120'] = cf.apply(lambda row: 1 if (row.b ==1) & (row.duration >= z_lower120) else 0, axis = 1)
    cf['missing120'] = cf.apply(lambda row: 1 if (row.duration > 120) & (row.duration <= upper120) else 0, axis = 1)

    # calculate missing
    sum60 = cf.loc[(cf.duration > 60) & (cf.duration <= upper60) & (cf.y_pred >= 0),'y_pred'].sum()
    sum120 = cf.loc[(cf.duration > 120) & (cf.duration <= upper120) & (cf.y_pred >= 0),'y_pred'].sum()

    # calculate probability of being in bin, avoiding negative probabilities
    cf['prob'] = 0
    cf.loc[(cf.duration > 60) & (cf.duration <= upper60) & (cf.y_pred >= 0), 'prob'] = cf.y_pred / sum60
    cf.loc[(cf.duration > 120) & (cf.duration <= upper120) & (cf.y_pred >= 0), 'prob'] = cf.y_pred / sum120

    print(cf.prob.sum() == 2)
    #print(cf[(cf.b60 == 1) | (cf.b120 == 1) | (cf.missing120 == 1) | (cf.missing60 == 1)])

    # save in dictionary
    bunching[f].update({'counterfactual_bins':cf})


# In[100]:


## difference in b60 need to be distributed to missing mass

# new column
data['duration_cf'] = data.duration_minutes

# in case of pooling etc.
notfirms = [x for x in bunching.keys() if x not in data.discom.unique()]
for x in notfirms:
    data[f'duration_cf_{f}'] = data.duration_minutes


for f in bunching:
    # subset dataframe but keep index
    if f in data.discom.unique():
        dt = data[data.discom == f]
        column = 'duration_cf'
    if f not in data.discom.unique():
        dt = data.copy()
        column = f'duration_cf_{f}'
    
    
    # get counterfactual bins
    cf = bunching[f]['counterfactual_bins']

    dict = {'60': cf.loc[(cf.b60 ==1) & (cf.difference >0),].index, '120': cf.loc[(cf.b120 ==1) & (cf.difference >0),].index}
    dict_m = {'60': cf.missing60, '120':cf.missing120}

    for d in ['60','120']:
        ind = dict[d]
        m = dict_m[d]
        for b in ind:
            # select bin
            bin = cf.bin[b]   
            # get 'extra' observations
            diff = np.round(cf.difference[b]).astype(int)
            #print(bin, diff)
            # extract the data in bin and select n = diff random values
            new_data = dt.loc[(dt.duration_cf >= bin[0]) & (dt.duration_cf <= bin[1])].sample(n = diff, axis = 0)
            # now randomly choose a bin in the missing mass, weighted by prob of observations being in that bin bin
            mis = cf.loc[m == 1,].sample(n=1, axis =0, weights = cf.prob).bin.reset_index(drop=True)[0]
            # create n=diff random values in bin
            new = np.random.poisson((mis[1] +  mis[0])/2, size = diff) # here I am assuming a poisson distribution in the bins
            # replace values in 
            data.loc[new_data.index.tolist(), column] = new
            dt.loc[new_data.index.tolist(), column] = new
    # update dictionary
    bunching[f].update({'data':dt})
           

#data.describe()


# ### plot of estimated counterfactual distribution and actual distribution

# In[96]:


# should look similar to counterfactual density from bunching estimation
fig, ax = plt.subplots(1,len(bunching),figsize=(15,5))
for i in range(len(bunching)):
    f = list(bunching.keys())[i]
    column = 'duration_cf'
    dt = data[data.discom == f]
    if f not in data.discom.unique(): 
        column = f'duration_cf_{f}'
        dt = data
    sns.kdeplot(dt[column][dt[column] < 200], label = 'counterfactual', ax = ax[i])
    sns.kdeplot(dt.duration_minutes[dt.duration_minutes < 200], label='observation', ax = ax[i])
    ax[i].axvline(60)
    ax[i].axvline(120)
    ax[i].legend()
    ax[i].set_title(f)
#plt.xlim(55,65)
#plt.ylim(0.000,0.004)


# In[97]:


# export 
data.to_csv(wd.parent/'data'/'data_transformed'/'outage_bunch.csv', index=False)

