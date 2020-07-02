#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 22:09:50 2020

@author: ramyagurunathan

Toberer Thermal Model: kappaL = A*(1 / T) + B
"""

import numpy as np
from math import pi
from scipy.optimize import curve_fit
import helper as hpr
import scipy.stats as ss
import os
import fnmatch
import pickle
import sys
import UnaryBayes.core_compute as cc
import UnaryBayes.core_plot as cp
import pandas as pd

 

'''
Methods for Property Conversions
'''

def v_sound(v_l, v_t):
    v_s = ((1 / (3 * v_l**3)) + (2 / (3 * v_t**3)))**(-1/3)
    return v_s

def debyeT(vs, atmV):
    return (hpr.hbar /hpr.kB) * (6*pi / atmV)**(1/3) * vs


#Can I incorporate the defect model into this..?
def kL_T(param, avgM, avgV, N, vs, T):
    '''
    kL_T = A(1/T) + B
    
    param = [coeff1, gruneisen, coeff2]
    '''
    A = (((6 * pi**2)**(2/3) / (4 * pi**2)) * param[0] * (avgM / hpr.Na) * 1e-3  * vs**3 / (param[1]**2 * avgV**(2/3))) * (N**(-1/3))
    B = param[2] * (3  * hpr.kB * vs / (2 * avgV**(2/3)))  * (pi / 6)**(1/3) * (1 - N**(-2/3))
    return A * (1 / T) + B

def gamma(stoich, og, subst, c):
    '''
    Calculate gamma for mass or strain
    '''
    natoms = sum(stoich)
    delP2 = 0
    denom = 0
    for n in range(len(og)):
        psite = subst[n]*c + og[n]*(1-c)        
        delP2 = delP2 + stoich[n]*c*(1-c)*(subst[n] - og[n])**2
        denom = denom + stoich[n]*psite               
    gamma = (delP2/natoms)/((denom/natoms)**2)
    return gamma 

def pd_tau(A, gamma, freq, avgV, vs):
    return A * 4 * pi * vs**3/ (avgV * freq**4 * gamma)

def umklapp_tau(B, grun2, freq, avgM, avgV, vs, T):
    return B * (6 * pi**2)**(1/3)/2 * ((avgM / hpr.Na) * 1e-3 * vs**3)\
 / (hpr.kB * avgV**(1/3) * float(grun2) * freq**2 * T) #grun2 is the squared gruneisen parameter

def spectral_C(vs, freq, T):
    x = hpr.hbar * freq / (hpr.kB * T)
    C = (3 / (2 * pi**2)) * hpr.kB * (freq**2/ vs**3) * (x**2 * np.exp(x))/ (np.exp(x) - 1)**2
    return C

def kL_umklapp_PD_vs_T(param, avgV, N, vs, stoich, og, subst, nfreq, c, T):
    '''
    param = [coeff1, coeff2, gruneisen, epsilon]
    '''
    gammaM = gamma(stoich, og['mass'], subst['mass'], c)
    gammaV = gamma(stoich, og['rad'], subst['rad'], c)
    gammatot = gammaM + param[3] * gammaV
    
    avgM = sum((1-c) * np.array(og['mass']) + c * np.array(subst['mass'])) / sum(stoich)
    debf = debyeT(vs, avgV) * (hpr.kB / hpr.hbar)
    dfreq = debf / nfreq
    kL = 0
    for freq in np.arange(dfreq, debf, dfreq):
        tauPD = pd_tau(param[0], gammatot, freq, avgV, vs)
        tauU = umklapp_tau(param[1], param[2], freq, avgM, avgV, vs, T)
        tau = 1/(tauU**(-1) + tauPD**(-1))
        kL = kL + (1/3) * spectral_C(vs, freq, T) * vs**2 * tau * dfreq
    return kL

def feval_klat(param, T, D):
    kL = kL_T(param, D['avgM'], D['avgV'], D['N'], D['vs'],T)
    return kL

def feval_klat_PD_U(param, T, D):
    kL = kL_umklapp_PD_vs_T(param, D['avgV'], D['N'], D['vs'], D['stoich'],\
                            D['og'], D['subst'], D['nfreq'], D['c'], T)
    return kL

#Can this just be added to the core_compute module?
def likelihood(param, D):
    dA = D['At'] - feval_klat_PD_U(param, D['Tt'], D)
    #Obtain hyperparameters for the zT data: this might be for scaling?
    #Try alternative dsitirubtions fro the log likelihood
    prob = ss.norm.logpdf(dA, loc=0, scale = param[-1]).sum()
    if np.isnan(prob):
        return -np.inf    
    return prob    


def read_data(data_dir, npts, dopant_conc : str):
    data = {}
    os.chdir(str(data_dir))
    try:
        os.chdir('Th Cond')
        os.chdir('F(temperature)')
        for file in os.listdir(os.getcwd()):
            if fnmatch.fnmatch(file, '*lathcond_' + dopant_conc + '.csv'):   
                d_array = np.genfromtxt(file, delimiter = ',')
                data['Th Cond'] = d_array[:npts]
        os.chdir('../..')
    except:
        raise hpr.PropertyError('Thermal conductivity data is unavailable or unformatted')     
    Tt = data['Th Cond'][:,0]
    At = data['Th Cond'][:,1]
    It = np.zeros(At.shape)
    Et = np.zeros(At.shape) #experimental error bars?
    return Tt, At, It, Et

def read_data_pd(data_dir, dopant_conc : str):
    '''
    Loop through each paper directory
    If desired dopant conc is a file in the directory, then store file as a dataframe
    Then, add dataframes to data arrays for UQ
    '''
    os.chdir(str(data_dir))
    data = {}
    subdirs = [x[0] for x in os.walk(data_dir)]
    Tt = []
    At = []
    try:
        for s in subdirs:
            os.chdir(s)
            for file in os.listdir(os.getcwd()):
                if fnmatch.fnmatch(file, '*x' + dopant_conc + '.txt'):
                    df = pd.read_csv(file, delimiter = '\t')
                    data = df.to_dict('list')
                    Tt += list(df['T(K)'])
                    At += data['kL (W/m/K)']
                else:
                    continue
    except:
        raise hpr.PropertyError('Data folders not formatted properly')
    Tt = np.array(Tt)
    At = np.array(At)
    It = np.zeros(At.shape)
    Et = np.zeros(At.shape)           
    return Tt, At, Et, It



if __name__ == '__main__':
            # for convenience, store all important variables in dictionary
        D = {}

        # save the current file name
        D['fname'] = sys.argv[0]

        # outname is the name for plots, etc
        D['outname'] = 'Ti_FeNbSb_klat'

        # set up a log file
        D['wrt_file'] = D['outname'] + '.txt'
        fil = open(D['wrt_file'], 'w')
        fil.close()
#        D['Tt'], D['At'], D['Et'], D['It'] = \
#        read_data('/Users/ramyagurunathan/Documents/PhDProjects/Fe2VAl/Fe2VAl/Data/Cobalt', 6, '20')
        D['Tt'], D['At'], D['Et'], D['It'] = \
        read_data_pd('/Users/ramyagurunathan/Documents/PhDProjects/Argonne_TECCA/UQData/FeNbSb/Titanium','20') 
        D['likelihood'] = likelihood
        
        D['name_list'] = ['Ti in FeNbSb']
        
        D['sampler'] = 'emcee'
        
        '''
        Define prior distributions. Do I need an 'epsilon' factor?
        '''
        D['distV'] = 4 * ['norm']
        #Parameters: [mstar, mob_param]
        D['locV'] = [1, 1, 1, 1] #centers of distributions
        D['scaleV'] = [.4, .4, .4, 0.4] #std. deviation of distributions
        D['dim'] = len(D['distV'])
        
        D['pname'] = ['A', 'B', 'gruneisen', 'epsilon' ]
        D['pname_plt'] = ['A', 'B', r'\gamma', r'\epsilon'] 
        
        D['n_param'] = 4
        
        '''
        Set material constants
        '''
#        Fe2VAl
        D['vs'] = v_sound(7750, 4530)
        D['avgM'] = 47.403
        D['avgV'] = 11.6e-30 #in cubic meters
        D['N'] = 4
        D['stoich'] = [1,1,1]
        D['nfreq'] = 1000
        D['og'] = {'mass': [55.845, 92.906, 121.76],'rad': [.75, .86, 0.9]}
        D['subst'] = {'mass': [47.867, 92.906, 121.76] ,'rad': [.75, .67, 0.9]}
        D['c'] = .2
        #FeNbSb
#        D['vs'] = 3052
#        D['avgV'] = (53.167E-30) / 3
#        D['avgM'] = 90.17
#        D['N'] = 3
        sampler_dict = {'nlinks' : 300, 'nwalkers' :50, 'ntemps' : 1, 'ntune' : 100}
        
        '''
        run MH algorithm to sample posterior
        '''
        D = cc.sampler_multip_emcee(D)
        with open(D['outname'] + '.pkl', 'wb') as buff:
            pickle.dump(D, buff)          
        
        '''
        remove the tuning samples from the raw trace
        (nwalkers, nlinks, dim)
        '''
        
        trace = D['rawtrace'][:, -D['nlinks']:, :]

        '''
        obtain a flattened version of the chain
        '''
        flattrace = trace.reshape((D['nlinks']*D['nwalkers'], len(D['pname'])))

        '''
        compute convergence diagnostics
        '''
        # Rhat (Gelman, 2014.) diagnoses convergence by checking the mixing
        # of the chains as well as their stationarity. Rhat should be less than
        # 1.1 for each variable of interest
        Rhat = cc.gelman_diagnostic(trace, D['pname'])
        msg = "Rhat: %s" % Rhat
        cc.WP(msg, D['wrt_file'])

        # neff (Gelman, 2014.) gives the effective number of samples for
        # each variable of interest. It should be greater than 10
        # for each variable
        neff = cc.effective_n(trace, D['pname'])
        msg = "effective sample size: %s" % neff
        cc.WP(msg, D['wrt_file'])        
        '''
        Plot Trace Information
        '''
        
        cp.plot_chains(D['rawtrace'], flattrace, D['nlinks'], D['pname'],
               D['pname_plt'], pltname=D['outname'])
#        cp.plot_squiggles(D['rawtrace'], 0, 1, D['pname_plt'], pltname=D['outname'])
        cp.plot_squiggles(D['rawtrace'], 0, 1, D['pname_plt'], pltname=D['outname'])
        
        '''
        Out Descriptions of Run
        '''
        msg = "sampling time: " + str(D['sampling_time']) + " seconds"
        cc.WP(msg, D['wrt_file'])
        
        #This is the marginal likelihood for each parameter
        msg = "model evidence: " + str(D['lnZ']) + \
              " +/- " + str(D['dlnZ'])
        cc.WP(msg, D['wrt_file'])
        
        param_true = None
        bounds = None
        '''
        Plot Results
        '''
        cp.plot_hist(flattrace, D['pname'], D['pname_plt'],
                 param_true=param_true, pltname=D['outname'])

        cc.coef_summary(flattrace, D['pname'], D['outname'])
    
        cp.plot_cov(flattrace, D['pname_plt'], param_true=param_true,
                    bounds=bounds, figsize=[5.5, 5.5], pltname=D['outname'])
    
#    
#    
        cp.plot_prediction(flattrace, D['name_list'],
                           D['Tt'], D['At'], D['It'], feval_klat_PD_U, D,
                           colorL=['k'], xlabel = 'T (K)', ylabel = r'$\kappa_L$ (W/m/K)')
            