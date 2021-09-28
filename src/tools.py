'''
*functions for the simulation and estimation
*classes and methods that can be used for the bunching estimation
'''

import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from sympy import symbols
from scipy import integrate
import statsmodels.api as sm


def C(x, sigma, alpha):
    '''
    calculate fixing cost
    '''
    first = sigma / alpha
    second = x / sigma
    power = -alpha
    return  first * (second**power)

def V(phi, phi120, x, N):
    '''
    fine to pay to customer
    '''
    if x <= 60:
        fine = 0
    elif x <= 120:
        fine = x * phi
    elif x > 120:
        fine = x* phi + phi120 * (x-120)
    return fine*N

def K(x,sigma, pi, phi, phi120, alpha, N):
    '''
    calculate total cost
    '''    
    if phi >0: return  pi * x + V(phi, phi120, x, N) + C(x, sigma, alpha)
    else: return pi * x + C(x, sigma, alpha)
    


def xopt(sigma, pi, phi, alpha, N):
    '''
    theoretical optimal x from minimization
    '''
    return sigma * ((pi + phi*N))**(-1/(1+alpha))
    

def solve_alpha(x, delta_x, pi, phi, phi120, startingvalue = 1):
    ### solve for alpha
    a = symbols('a')
    
    def sigmaI(a):
        '''
        sigma as a function of x
        '''
        if x == 60: z = (x + delta_x)*(pi)**(1/(1+a))
        if x == 120: z = (x + delta_x)*(pi + phi)**(1/(1+a))
        return z

    def  xI(a):
        '''
        x with fine
        '''
        if x == 60: s = sigmaI(a) * (pi+phi)**(-1/(1+a))
        if x == 120: s = sigmaI(a) * (pi+phi120)**(-1/(1+a))
        return  s

    def L(a):
        '''
        cost under fine
        '''
        return K(xI(a), sigmaI(a), pi, phi, phi120,alpha=a, N=1)        

    def R(a):
        '''
        cost when bunching
        '''
        if x == 60: y = pi*x + C(x, sigmaI(a), a)
        if x == 120: y = (pi + phi)*x + C(x, sigmaI(a), a)
        return  y

    def solution(a):
        '''
        when equality holds
        '''
        return L(a) - R(a)

    return fsolve(solution, startingvalue)[0]
    

class bunching:

    def __init__(self, data, bsize,xmax, xmin, z_upper, z_lower, missing, ex_reg, ex_reg_miss, poly_dgr, include_missing = True):

        '''
        data: data - series to be used
        xcol: string with the column name of the observation where bunching happens
        bsize: binsize
        xmax: maximum number of x's to be taken into account
        xmin: minimum number of x's to be taken into account
        z_upper, z_lower: defines regions around bunching happens
        missing: up to which x we expect missing mass
        ex_reg: number of bins to exclude around bunching
        poly_dgr: degree of polynomial to be used in the estimation
        '''
       
        self.x = data     
        self.bsize = bsize
        self.xmax = xmax
        self.xmin = xmin
        self.z_upper = z_upper
        self.z_lower = z_lower
        self.missing = missing
        self.ex_reg = ex_reg
        self.ex_reg_miss = ex_reg_miss
        self.poly_dgr = poly_dgr
        self.incl_missing = include_missing


    def df_count(self):
        '''
        creates the dataframe by forming bins
        '''
        nbins = int(self.xmax/self.bsize +2)
        bins = [(x) * self.bsize  for x in range(nbins)]

        nobs = self.x.groupby(pd.cut(self.x, bins = bins, right = False)).count()       

        # put it in a df
        df_count = pd.DataFrame(list(zip([(x,x+self.bsize) for x in bins],nobs)), columns=['bin', 'nobs'])

        df_count['duration'] = df_count.apply(lambda row: row.bin[0], axis = 1)
        df_count = df_count.loc[(df_count.duration < self.xmax) & (df_count.duration >= self.xmin), ].reset_index(drop=True)
        return df_count

    def create_vars(self):
        '''
        make polynomials and the bunching and missing mass
        '''
        df_count = self.df_count()     
        # make polynomials 
        for i in range(2,self.poly_dgr +1):
            n = 'duration' + str(i)
            df_count[n] = df_count.duration ** i
        # add dummy vars
        df_count['b'] = ((df_count.duration <= self.z_upper) & (df_count.duration >= self.z_lower)).astype(int)
        df_count['m'] = ((df_count.duration > self.z_upper) & (df_count.duration <= self.missing)).astype(int)

        return df_count

    def get_coefs(self):
        '''
        coefficients to be used in the estimation
        '''
        df_count = self.create_vars()
        df_count = df_count.assign(Intercept = 1)
        # get columns
        coefs = ['Intercept']
        coefs.append('duration')
        [coefs.append('duration' + str(i)) for i in range(2,self.poly_dgr + 1)]
        coefs.append('b')
        if self.incl_missing == True:
            coefs.append('m')
        X = df_count.loc[:,coefs]
        return X

    def estimation(self):
        '''
        bunching estimation
        '''
        df_count = self.df_count()
        X = self.get_coefs()
        y = df_count.loc[:,'nobs']
        model = sm.OLS(y , X).fit()
        return model

    def estimation_res(self):
        '''
        gives out the results from the bunching estimation
        '''
        model = self.estimation()
        return model.summary()

    def prediction(self):
        '''
        sets bunching and missing dummies to zero and makes prediction
        '''
        df_count = self.create_vars()
        X = self.get_coefs()
        X.b = 0
        X.m = 0
        model = self.estimation()
        df_count['y_pred'] = model.predict(X)
        return df_count

    def get_deltaX(self):
        '''
        calculates and returns delta x
        '''
        df = self.prediction()
        x = np.sum(df.loc[(df.duration <= self.z_upper) & (df.duration > self.z_lower), 'y_pred'])
        y = np.sum(df.loc[(df.duration <= self.z_upper) & (df.duration > self.z_lower), 'nobs'])
        excess_x = (y-x) /(x/self.ex_reg)
        return excess_x
   
    def get_B(self):
        '''
        calculates and returns Bunching Mass
        '''
        df = self.prediction()
        x = np.sum(df.loc[(df.duration <= self.z_upper) & (df.duration > self.z_lower), 'y_pred'])
        y = np.sum(df.loc[(df.duration <= self.z_upper) & (df.duration > self.z_lower), 'nobs'])
        B = (y-x)
        return B
   
    def get_mX(self):
        '''
        calculates and returns missing
        '''
        df = self.prediction()
        xm = np.sum(df.loc[(df.duration > self.z_upper) & (df.duration <= self.missing), 'y_pred'])
        ym = np.sum(df.loc[(df.duration > self.z_upper) & (df.duration <= self.missing), 'nobs'])
        m_x = (ym-xm)/(xm/self.ex_reg_miss)
        return m_x

    def total_bunch(self):
        '''
        calculates and returns total bunching
        '''
        model = self.estimation()
        df = self.prediction()
        x = np.sum(df.loc[(df.duration <= self.z_upper) & (df.duration > self.z_lower), 'y_pred'])
        y = np.sum(df.loc[(df.duration <= self.z_upper) & (df.duration > self.z_lower), 'nobs'])
        def  I(r):
            add = 0
            for d in range(self.poly_dgr +1):
                add += model.params[d]* r**d
            return add

        def gap(d):
            return (y -x) - integrate.quad(I,60,60+d)[0]

        res = fsolve(gap, 35)*self.bsize
       
        return res[0]