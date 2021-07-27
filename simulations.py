'''
simulations of model
'''

import numpy as np
import pandas as pd
from sympy import symbols, solve
from scipy.optimize import fmin
import math
import matplotlib.pyplot as plt
import seaborn as sns

# draw randon number from lognormal distribution
n = 1000
sigma = np.random.lognormal(size=n)
# make to list
sigma = sigma.tolist()
# define pi
pi = 1
phi = 20


# define function C
def C(sigma, x):
    return sigma/x

# define function K
def K(x,sigma, pi=1, phi=0):
    c = C(sigma, x)
    exp = (pi + phi) * x + c
    return math.exp(exp)

# solve for x based on sigma
xstar = {}
for s in sigma:
    x = fmin(K,0.1, args=(s,pi, 0), disp=False)
    xstar[s] = x[0]

# now add parameter phi
xstar_phi = {}
for s in sigma:
    x = fmin(K,0.1, args=(s,pi, phi), disp=False)
    xstar_phi[s] = x[0]

# put in a df
df = pd.DataFrame.from_dict(xstar, orient='index', columns=['xstar'])
df = df.reset_index()
df = df.rename(columns={'index':'sigma'})

df_phi = pd.DataFrame.from_dict(xstar, orient='index', columns=['xstar_phi'])
df_phi = df_phi.reset_index()
df_phi = df_phi.rename(columns={'index':'sigma'})

# merge to one
df = df.merge(df_phi)

# calculate fixing cost C
df['fixing_cost'] = df.apply(lambda row: C(row.sigma,row.xstar), axis=1)
# calculate cost K
df['K'] = df.apply(lambda row: K(row.xstar, row.sigma), axis=1)

# calculate fixing cost C
df['fixing_cost_phi'] = df.apply(lambda row: C(row.sigma,row.xstar_phi), axis=1)
# calculate cost K
df['K_phi'] = df.apply(lambda row: K(row.xstar_phi, row.sigma), axis=1)

# make plots of x, C, K
fig, axs = plt.subplots(3,figsize=(15,15))
axs[0].plot(df.sigma, df.xstar, 'o')
axs[0].set_title('xstar')
axs[1].plot(df.sigma, df.fixing_cost, 'o')
axs[1].set_title('fixing cost')
axs[2].plot(df.sigma, df.K, 'o')
axs[2].set_title('K')
plt.show()

# plot tradeoff
left = [60*pi + s/60 for s in sigma]
right = [np.sqrt(2*(s*(pi+phi))) for s in sigma]

plt.plot(sigma, left, 'o', label = 'left')
plt.plot(sigma, right, 'o', label = 'right')
plt.legend()

# density plot of sigmas
sns.kdeplot(df.xstar)


