#!/usr/bin/env python
# coding: utf-8

# # bunching estimation using the real data
# 
# take concept from simulation

# In[1]:


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


# In[2]:


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
data.describe()


# $\pi$: revenue loss from outage
# 
# Calculation: \
# $r =$ unserved kWh / duration of outage \
# $\pi = r * tariff$
# 
# unserved kWh is a function of the number of customers affected and the duration of the outage

# In[3]:


## calculate pi 

#the tariff is a price per kwh
tariff = 8

data['r_permin'] = data.apply(lambda row: row.unservedmuduetooutage * 1e6 / row.duration_minutes, axis = 1)

data['pi'] = data.r_permin * tariff

pi = data.pi.median()

print('median pi:', pi)



# define parameters 
phi = 50/60 
phi120 = 100/60
print(phi, phi120)

sns.kdeplot(data.r_permin)


#data.r_permin.describe()


# In[4]:


# 
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].scatter(data.hour_of_day, data.unservedmuduetooutage, label = 'unservedMU')
ax[1].scatter(data.hour_of_day, data.r_permin, label='r_permin', color='red')
fig.legend()


# density of duration 

# In[5]:


xmax  = 420

fig, axs = plt.subplots(1,2,figsize=(18,5))
sns.kdeplot(data.duration_minutes, ax = axs[0])
axs[0].axvline(60, color  = 'grey')
axs[0].axvline(120, color  = 'grey')
axs[0].set_xlim(0,xmax)

axs[1].hist(data.duration_minutes, histtype = 'step', bins = xmax)
axs[1].axvline(60, color  = 'grey')
axs[1].axvline(120, color  = 'grey')
axs[1].set_xlim(0,xmax)


# # Bunching Estimation

# In[6]:


# bunching at x = 60
bsize = 1
ex_reg = 10
ex_reg_miss = 20
z = 59
z_lower60 = z - bsize*ex_reg

missing60 = z + ex_reg_miss*bsize

bunch60 = tools.bunching(data.duration_minutes, bsize = bsize, xmax= 115, xmin= 0, z_upper= z, z_lower= z_lower60, missing = missing60, ex_reg= ex_reg, ex_reg_miss=ex_reg_miss, poly_dgr=6)
#print(bunch60.df_count())

display(bunch60.estimation_res())

print('EX:', bunch60.get_deltaX(), 'mX:', bunch60.get_mX(), 'B:', bunch60.get_B())

pred60 = bunch60.prediction()

print('total bunching',bunch60.total_bunch())


# In[7]:


# bunching at x = 120
bsize = 1
ex_reg = 10
z = 119
z_lower120 = z - 5 * bsize
missing120 = z + ex_reg * bsize

# define bins

bunch120 = tools.bunching(data.duration_minutes, bsize = bsize, xmax= 160, xmin= 90, z_upper= z, z_lower= z_lower120, missing = missing120, ex_reg= ex_reg, ex_reg_miss = ex_reg_miss, poly_dgr=9, include_missing=False)
#print(bunch120.df_count())

display(bunch120.estimation_res())

print('deltaX:', bunch120.get_deltaX())

pred120 = bunch120.prediction()

print('total bunching',bunch120.total_bunch())


# plot of counterfactual and actual data

# In[8]:


# concat the two predictions
pred = pd.concat([pred60,pred120], ignore_index=True)
#xs = tools.bunching(data.duration_minutes, bsize = bsize, xmax= xmax, xmin= 0, z_upper= z, z_lower= z_lower120, missing = missing120, ex_reg= ex_reg, ex_reg_miss = ex_reg_miss, poly_dgr=6, include_missing=False).df_count()


fig, ax = plt.subplots(figsize=(7,5))
plt.plot(pred.duration, pred.y_pred)
plt.plot(pred.duration, pred.nobs)
plt.axvline(60, color= 'grey', linestyle = 'dashed')
plt.axvline(120, color= 'grey', linestyle = 'dashed')
plt.xticks(ticks = [60, 120])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('Duration (Minutes)')


# Visualize $\Delta x$

# In[9]:


fig, ax = plt.subplots()
b = np.round(60 + bunch60.total_bunch())
min = -300
plt.plot(pred60.duration[pred60.duration >= 0], pred60.y_pred[pred60.duration >= 0], color = 'black', linestyle='dashed', label = 'counterfactual')
plt.plot(pred60.duration, pred60.nobs, color = 'violet')
plt.plot([60,60], [min,pred60.loc[pred60.duration == 60, 'y_pred'].reset_index(drop=True)[0]], color = 'grey')
plt.axvline(b,color='grey', linestyle = 'dashed')
plt.xlabel('Duration (Minutes)')
plt.ylabel('# observations')
plt.xticks([60 , b], labels = [60,b])
plt.ylim(min,pred60.nobs.max()+ 100)
plt.fill_between(np.linspace(60,b,np.sum((pred60.duration > 60) & (pred60.duration < b))),pred60.y_pred[(pred60.duration > 60) & (pred60.duration < b)], np.repeat(min,np.sum((pred60.duration > 60) & (pred60.duration < b))), alpha = .3, color = 'grey')
plt.annotate('bunching mass', xy = (67,min + 50), size = 7)
plt.legend()

#fig.savefig('xdensity.png',dpi=120, format='png')


# ## get counterfactual duration at outage level
# 
# !! this is maybe not optimal. Right now, I am assuming a poisson distribution in the bins of the missing mass. \
# 

# In[10]:


# first need to get the counterfactuals on the outage-level
cf = pd.concat([pred60,pred120], ignore_index=True)
cf = cf.loc[:,['bin', 'nobs', 'duration', 'b', 'm','y_pred']]

cf['difference'] = cf.nobs - cf.y_pred
# make sure that at least one bin is in missing
upper60 = 60 + bunch60.total_bunch() + bsize 
upper120 = 120 + bunch120.total_bunch() + bsize 
# create dummies
cf['b60'] = cf.apply(lambda row: 1 if (row.b ==1) & (row.duration <= missing60) else 0, axis = 1)
cf['missing60'] = cf.apply(lambda row: 1 if (row.duration > 60) & (row.duration <= upper60) else 0, axis = 1)
cf['b120'] = cf.apply(lambda row: 1 if (row.b ==1) & (row.duration >= z_lower120) else 0, axis = 1)
cf['missing120'] = cf.apply(lambda row: 1 if (row.duration > 120) & (row.duration <= upper120) else 0, axis = 1)

# calculate missing
sum60 = cf.loc[(cf.duration > 60) & (cf.duration <= upper60),'y_pred'].sum()
sum120 = cf.loc[(cf.duration > 120) & (cf.duration <= upper120),'y_pred'].sum()

# calculate probability of being in bin
cf['prob'] = 0
cf.loc[(cf.duration > 60) & (cf.duration <= upper60), 'prob'] = cf.y_pred / sum60
cf.loc[(cf.duration > 120) & (cf.duration <= upper120), 'prob'] = cf.y_pred / sum120

print(cf.prob.sum() == 2)
cf[(cf.b60 == 1) | (cf.b120 == 1) | (cf.missing120 == 1) | (cf.missing60 == 1)]


# In[11]:



## difference in b60 need to be distributed to missing mass
## can we improve this? It is okay but not optimal

data['duration_cf'] = data.duration_minutes

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
        new_data = data.loc[(data.duration_cf >= bin[0]) & (data.duration_cf <= bin[1])].sample(n = diff, axis = 0)
        # now randomly choose a bin in the missing mass, weighted by prob of observations being in that bin bin
        mis = cf.loc[m == 1,].sample(n=1, axis =0, weights = cf.prob).bin.reset_index(drop=True)[0]
        # create n=diff random values in bin
        new = np.random.poisson((mis[1] +  mis[0])/2, size = diff) # here I am assuming a poisson distribution in the bins
        # replace values in new_data
        data.loc[new_data.index.tolist(),'duration_cf'] = new

#data.describe()


# In[12]:


# should look similar to counterfactual density from bunching estimation
sns.kdeplot(data.duration_cf[data.duration_cf < 200], label = 'cf')
sns.kdeplot(data.duration_minutes[data.duration_minutes < 200], label='data')
#plt.xlim(55,65)
#plt.ylim(0.000,0.004)
plt.legend()

plt.axvline(60)
plt.axvline(120)


# In[14]:


# export 
data.to_csv(wd.parent/'data'/'data_transformed'/'outage_bunch.csv', index=False)

