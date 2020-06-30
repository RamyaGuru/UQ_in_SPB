#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:53:18 2020

Single Parabolic Band Model implemented for UQ

@author: ramyagurunathan
"""

import numpy as np
from fdint import fdk
from math import pi, exp
import helper as hpr
import scipy.stats as ss
import os
import sys
import fnmatch
import pickle
import UnaryBayes.core_compute as cc
import UnaryBayes.core_plot as cp
import matplotlib.pyplot as plt
import pandas as pd

np.seterr(divide='ignore', invalid="ignore")

'''
Single Parabolic Band Methods

Rather than using sigma_e0 directly, use sigma_e0 = mob_param * (m*)^(3/2)

Here, m* is the Seebeck effective mass while the inertial effective mass is 
within the mob_param
'''

def sigmae0_from_muW(muW, T):
    return (8 * pi * hpr.e * (2 * hpr.me * hpr.kB * T)**(3/2) / (3 * hpr.h**3)) * muW

def carr_conc_from_eta(eta, mstar, T=300):
    return 4 * pi * (2 * mstar * hpr.me * hpr.kB * T / hpr.h**2)**(3/2) * fdk(1/2, eta)

def seebeck(eta):
    return (hpr.kB / hpr.e) * (2 * fdk(1, eta) / fdk(0, eta) - eta)

def lorentz(eta):
    return (hpr.kB**2 / hpr.e**2) * (3 * fdk(0, eta) * fdk(2, eta) - 4 * fdk(1, eta)**2)\
/(fdk(0,eta)**2)

def conductivity(eta, T, mob_param, mstar, s=1):
    sigmae0 = sigmae0_from_muW(mob_param * mstar**(3/2), T)
    if s == 0:  # s=0 requires analytic simplification
        return sigmae0/ (1. + np.exp(-eta))
    else:
        return sigmae0 * s * fdk(s - 1, eta)


#Maybe for now provide list of lattice thermal conducitvities
def zT_from_eta(mstar, mob_param, eta, kL, T = [300]):
#    print(mstar, mob_param) #eta should also be a fitting parameter..??
    S = seebeck(eta)
    cond = conductivity(eta, T, mob_param, mstar, s = 1)
    L = lorentz(eta)
    kappa_e = L * cond * T
    return S**2 * cond * T / (kappa_e + kL)

    

'''
Constant Temperature: zT versus n
'''
kL = 1 #W/m/K
T = 300
def feval_zT(param, T, D):
    zT = zT_from_eta(*param, D['eta'], D['kL'], T = T)
    return zT
    
def likelihood(param, D):
    dA = D['At'] - feval_zT(param, D['Tt'], D)
    #Obtain hyperparameters for the zT data: this might be for scaling?
    prob = ss.norm.logpdf(dA, loc=0, scale = param[-1]).sum() 
    return prob



class PropertyError(Exception):
    def __init__(self, message):
        self.message = message

def read_data(data_dir, npts, dopant_conc : str):
    data = {}
    os.chdir(str(data_dir))
    try:
        os.chdir('Th Cond')
        os.chdir('F(temperature)')
        for file in os.listdir(os.getcwd()):
            if fnmatch.fnmatch(file, '*_' + dopant_conc + '.csv'):   
                d_array = np.genfromtxt(file, delimiter = ',')
                data['Th Cond'] = d_array[:npts]
        os.chdir('../..')
    except:
        raise PropertyError('Thermal conductivity data is unavailable or unformatted')
    try:
        os.chdir('Seebeck')
        os.chdir('F(temperature)')
        for file in os.listdir(os.getcwd()):
            if fnmatch.fnmatch(file, '*_' + dopant_conc + '.csv'): 
                d_array = np.genfromtxt(file, delimiter = ',')
                data['Seebeck'] = d_array[:npts]
        os.chdir('../..')
    except:
        raise PropertyError('Seebeck data is unavailable or unformatted')   
    try:
        os.chdir('Conductivity')
        os.chdir('F(temperature)') 
        for file in os.listdir(os.getcwd()):
            if fnmatch.fnmatch(file, '*_' + dopant_conc + '.csv'): 
                d_array = np.genfromtxt(file, delimiter = ',')
                data['Conductivity'] = d_array[:npts]
        os.chdir('../..')
    except:
        raise PropertyError('Electrical Conductivity data is unavailable or unformatted')
    '''
    Following the notation of Noah: Tt (indepedent), At (dependent), It (iterator?), Et (error?)
    '''
    Tt = data['Th Cond'][:,0]
    At = ((data['Seebeck'][:,1] * 1e-6)**2 * data['Conductivity'][:,1] * Tt) / (data['Th Cond'][:,1])
    It = np.zeros(At.shape)
    Et = np.zeros(At.shape)
    return Tt, At, Et, It

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
                    At += [s**2 * c * T / k for s,c,k,T in zip(data['S (V/K)'],\
                                                               data['sigma (S/m)'], 
                                                               data['kappa (W/m/K)'],
                                                               data['T(K)'])]
                else:
                    continue
    except:
        raise PropertyError('Data folders not formatted properly')
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
        D['outname'] = 'Fe2VAl_SPB'

        # set up a log file
        D['wrt_file'] = D['outname'] + '.txt'
        fil = open(D['wrt_file'], 'w')
        fil.close()
        D['Tt'], D['At'], D['Et'], D['It'] = \
        read_data_pd('/Users/ramyagurunathan/Documents/PhDProjects/Argonne_TECCA/UQData/FeNbSb/Titanium', '20')
#        D['Tt'], D['At'], D['Et'], D['It'] = \
#        read_data('/Users/ramyagurunathan/Documents/PhDProjects/Fe2VAl/Fe2VAl/Data/Cobalt/', 6, '50')
        
        D['likelihood'] = likelihood
        
        D['name_list'] = ['Co in Fe2VAl']
        
        D['sampler'] = 'emcee'
        
        '''
        Define prior distributions. Do I need an 'epsilon' factor?
        '''
        D['distV'] = 2 * ['norm']
        #Parameters: [mstar, mob_param
#        D['s'] = [.1, .1]
        D['locV'] = [1.5, 200e-4] #centers of distributions: [1.5, 200e-4]
        D['scaleV'] = [.3, 40e-4] #std. deviation of distributions: [0.3, 40e-4]
        D['dim'] = len(D['distV'])
        
        '''
        Initalize property traces and trace plots
        '''
        D['pname'] = ['mstar', 'mob_param']
        D['pname_plt'] = ['m^*', r'\mu_0']
        
        D['n_param'] = 2
        
        '''
        Iniitalize eta and kL as constants
        '''
        D['kL'] = 6
        D['eta'] = 1 #need to adjust this?? should this also be a parameter..??
        
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
                           D['Tt'], D['At'], D['It'], feval_zT, D,
                           colorL=['k'], xlabel = 'T (K)', ylabel = 'zT')
        
#    
