#!/usr/bin/env python
# coding: utf-8

# In[36]:


import tools
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols
from scipy.optimize import fsolve
import seaborn as sns
import numpy as np
from lmfit import Parameters, fit_report, minimize
get_ipython().system('jupyter nbconvert --to script outage_bunching.ipynb ')
# converts notebook to py file needed to import results from bunching
from outage_bunching import bunch60, bunch120


np.random.seed(111)


# In[26]:


dx60 = bunch60.total_bunch()
dx120 = bunch120.total_bunch()
print(dx60, dx120)


# In[27]:


wd = Path.cwd()
data = pd.read_csv(wd.parent/'data'/'data_transformed'/'outage_bunch.csv')

data.head()


# $\pi$: revenue loss from outage
# 
# Calculation: \
# $r =$ unserved kWh / duration of outage \
# $\pi = r * tariff$
# 
# unserved kWh is a function of the number of customers affected and the duration of the outage

# In[6]:


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

#sns.kdeplot(data.r_permin)

# 
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].scatter(data.hour_of_day, data.unservedmuduetooutage, label = 'unservedMU')
ax[1].scatter(data.hour_of_day, data.r_permin, label='r_permin', color='red')
fig.legend()

#data.r_permin.describe()


# ## model estimation : alpha

# non-linear least squares

# $ (\frac{\pi + \phi}{\pi}) (\frac{x}{x + \Delta x}) = \frac{1}{\alpha} [(\frac{\pi + \phi}{\phi})^{\alpha / 1+\alpha} (1+\alpha) - (1 + \frac{\Delta x}{x})^{\alpha}]$

# In[149]:


# code source: https://lmfit.github.io/lmfit-py/fitting.html

data_clean = data[data.pi >0] # keep only non-negative values
# calculate left side of equation
left = data_clean.apply(lambda row: (row.pi + phi) / row.pi * 60/(60+dx60) , axis = 1 ).to_numpy()

pars = Parameters()
pars.add('alpha', value = .8)

def residual(pars,pi, data ,phi = phi, x = 60, dx = dx60):
    vals = pars.valuesdict()
    alpha = vals['alpha']
    # RHS of equation
    model = 1/alpha * (((pi + phi)/phi)**(alpha /(1+alpha)) * (1+alpha) - (1 + (dx/ x))**alpha )
    return model - data

fit_params = Parameters()
fit_params.add('alpha', value = .5)

# fit model
out = minimize(residual, fit_params, args = (data_clean.pi.to_numpy() ,left))
print(fit_report(out)) # output


# In[168]:


print('-------------------------------')
print('Parameter    Value       Stderr')
for name, param in out.params.items():
    print(f'{name:7s} {param.value:11.5f} {param.stderr:11.5f}')


# In[170]:


alpha_ls = out.params['alpha'].value


# 
# 
# solve for $\alpha$ and compare: $K(x^I) = K(x^*)$
# 
# $K(x^I) = (\pi + \phi)^{\alpha /1+\alpha} (x^* + \Delta x^*) \pi^{1/1+\alpha}  (1 + 1/\alpha)$
# 
# 
# at $x$ around $60$: $\phi = 50$
# 
# If $x>120$: $\phi = 100$
# 
# 
# I am now doing it for N=1, not sure if that's right. Also, we should double check the functions. What values of $\pi$ and $\phi$ should we use? For $\pi$ we could maybe do a weighted average

# In[28]:


alpha60 = tools.solve_alpha(60, bunch60.total_bunch(), pi, phi=50/60, phi120 = 100/60, startingvalue = 10)
alpha120 = tools.solve_alpha(120, bunch120.total_bunch(), pi, phi=50/60, phi120 = 100/60, startingvalue=10)

print('alpha:',alpha60, 'deltax', bunch60.total_bunch())
print('alpha:',alpha120, 'deltax', bunch120.total_bunch())

alpha = alpha60
#alpha = 0.8

# plot indifference as a function of alpha
delta_x = bunch60.total_bunch()
def sigmaI(a):
    return (60 + delta_x)*(pi)**(1/(1+a))

def  xI(a):
    return  sigmaI(a) * (pi+50/60)**(-1/(1+a))

def L(a):
    return tools.K(xI(a), sigmaI(a), pi=pi, phi=50/60, phi120 = 100/60,alpha=a, N=1)
    
def R(a):
    return pi*60 + tools.C(60, sigmaI(a), a) 

def solution(a):
    return L(a) - R(a)

alphas = np.linspace(0,15, 100)
sol =  [solution(a) for a in alphas]
plt.plot(alphas,sol)
plt.axhline(0, linestyle = 'dashed')
plt.axvline(alpha60, linestyle='dashed', color = 'red')


# ## calculate the $\sigma$'s based on the counterfactual x's and alpha's

# In[185]:


data['sigma'] = 0

def get_sigma(s,x, N, pi):
    if x < 60:
        exp = tools.xopt(s, pi, phi=0, alpha = alpha, N=N) - x
    elif x < 120:
        exp = tools.xopt(s, pi, phi=50/60, alpha = alpha, N=N) - x
    else:   
        exp = tools.xopt(s, pi, phi=100/60, alpha = alpha, N=N) - x
    return exp

for i in data.index:
    s = symbols('s')
    x = data.loc[i, 'duration_cf']
    N = data.loc[i, 'noofcustomersaffected']
    pi = data.loc[i, 'pi']
    data.loc[i,'sigma'] = fsolve(get_sigma, x0 = 10, args = (x, N, pi))[0]


# In[183]:


sns.kdeplot(x = data.sigma)
#plt.xlim(-1000,20000)
data.sigma.describe()


# In[186]:


# export 
data.to_csv(wd.parent/'data'/'data_transformed'/'outage_bunch_est.csv', index=False)

