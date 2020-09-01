#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:24:52 2020

@author: ramyagurunathan

single parabolic band--> new scattering mechanisms

Script redefines the SPB functions to utilize the scattering coefficients

Fit to Pisarenko Plots

--> do fit to Seebeck and conductivity versus doping level separately?

Scattering coefficient s:
    l = 0: acoustic phonon scattering
    l = 0.5: neutral impurity scattering
    l = 1: optical phonon scattering
    l = 2: ionized impurity scattering
    
    
Note: for non-APS scattering, the tau_0

Also add sequential fitting to the Seebeck and lattice thermal conductivity?

Code so that independent variable can be carrier conc or temperature?
"""


import single_parabolic_band as spb
import numpy as np
import helper as hpr
from fdint import fdk
from math import pi
import os
import pandas as pd
import klat_temperature as kL
import scipy.stats as ss
from scipy.optimize import minimize
import UnaryBayes.core_compute as cc
import UnaryBayes.core_plot as cp
import pickle

np.seterr(divide='ignore', invalid="ignore")

def carrconc_from_dopantconc(x, D):
    '''
    x is a dopant; will assume it contributes 1 carrier and scatters
    more or less like Ge
    '''
    return x * D['deg'] * D['Na'] * (1/D['mol_mass']) * (D['density'] * 1e3) #D['deg'] is the site degenracy of the defect
    
def dopant_conc_from_carrconc(n, D):
    return n * (1 / (D['deg'] * D['Na'])) * (D['mol_mass']) * (1/(D['density'] * 1e3))


def wtd_mobility(coeff, mstar, T, l = 0):
    if l == 0 or l ==1: #any time it is a lattice vibration?
        n = l - (3/2)
    else: 
        n = l - (1/2)
    return (coeff * (1 / mstar)) * (T**(n))

def sigmae0_from_muW(muW, T):
    return spb.sigmae0_from_muW(muW, T)

def conductivity_s(eta, T, coeff, mstar, l = 0):
    muW = wtd_mobility(coeff, mstar, T, l)
    sigmae0 = sigmae0_from_muW(muW, T)
    if l == -1:  # s=0 requires analytic simplification
        return sigmae0/ (1. + np.exp(-eta))
    else:
        return sigmae0 * (l + 1) * fdk(l, eta)
    
def conductivity_s_simp(eta, T, muW, l = 0):
    sigmae0 = sigmae0_from_muW(muW, T)
    if l == -1:  # s=0 requires analytic simplification
        return sigmae0/ (1. + np.exp(-eta))
    else:
        return sigmae0 * (l + 1) * fdk(l, eta)    
    
def seebeck_s(eta, l = 0):
    return (hpr.kB / hpr.e) * ((2 + l) / (1 + l) * (fdk(l + 1, eta)/ fdk(l, eta)) - eta)

def lorenz_s(eta, l = 0):
    return (hpr.kB / hpr.e)**2 * (1 + l) * (3 + l) * fdk(l, eta) * fdk(l + 2, eta) -\
(2 + l)**2 * fdk(l + 1, eta)**2 / ((1 + l)**2 * fdk(l, eta)**2)

#def carr_conc_s(eta, mstar, T = 300):
#    return spb.carr_conc_from_eta(eta, mstar, T)

'''
Calculation of RH -> will return in units of cm^3/C
'''
def RH_s(eta, mstar, T = 300, l =0):
    return (3 * hpr.hbar**3 * (2 * pi)**3 / (8 * pi * hpr.e * (2 * mstar * hpr.me * hpr.kB * T)**(3/2)) *\
((1/2) + 2 * l) * fdk(2 * l - (1/2), eta) / ((1 + l)**2 * fdk(l, eta)**2)) * (1e6)

def carr_conc_s(eta, mstar, T = 300, l = 0):
    return 1 / (RH_s(eta, mstar, T, l) * hpr.e)


def carr_conc_from_Seebeck(mstar, S, T = 300, l = 0):
    eta = []
    for s in S:
        eta.append(minimize(
            lambda eta: np.abs(seebeck_s(eta, l) - s),
            method='Nelder-Mead', x0=[0.]).x[0])
    return carr_conc_s(np.array(eta), mstar, T, l)

'''
Jonker plot methods: returns conductivity in units of S/cm
'''
def conductivity_from_Seebeck(muW, S, T = 300, l = 0):
    eta = []
    for s in S:
        eta.append(minimize(
            lambda eta: np.abs(seebeck_s(eta, l) - s),
            method='Nelder-Mead', x0=[0.]).x[0])
    return conductivity_s_simp(np.array(eta), T, muW, l) * 1E-2

'''
Mu_0 forms
'''


def kL_from_eta(kL_param, eta, mstar, kL_in: dict, T): 
    n = carr_conc_s(eta, mstar, T)
    c = dopant_conc_from_carrconc(n)    
    return kL.kL_umklapp_PD_vs_T(kL_param, kL_in['avgV'], kL_in['N'],\
                               kL_in['vs'], kL_in['stoich'], kL_in['og'],\
                               kL_in['subst'], kL_in['nfreq'], c, T)

def zT_from_eta(eta, mstar, coeff, sigmae0, kL_param, kL_in, T = 300, l = 0):
    S = seebeck_s(eta) # should just fit to the Seebeck coefficient
    cond = conductivity_s(eta, T, coeff, mstar, l)
    L = lorenz_s(eta)
    kappa_e = L * cond * T
    n = carr_conc_s(eta, mstar, T)
    c = dopant_conc_from_carrconc(n)
    kappaL = kL.kL_umklapp_PD_vs_T(kL_param, kL_in['avgV'], kL_in['N'],\
                               kL_in['vs'], kL_in['stoich'], kL_in['og'],\
                               kL_in['subst'], kL_in['nfreq'], c, T)
    return S**2 * cond * T / (kappa_e + kappaL)


def read_data_pd(data_dir, name_list):
    os.chdir(str(data_dir))
    data = {}
    try:
        for file in os.listdir(os.getcwd()):
            if str(file) in name_list:
                print(str(file))
                df = pd.read_csv(file, delimiter = '\t')
                data[str(file)] = df.to_dict('list')
    except:
        raise hpr.PropertyError('Data folders not formatted properly')  
    return data

def get_data(data : dict, ind_prop : str, dep_prop : str, error = 0.15):
    Tt = []
    At = []
    Et = []
    It = []
    i = 0
    for k, v in data.items():
        print(v.keys())
        Tt += v[ind_prop]
        At += v[dep_prop]
        Et += v[dep_prop]
        It +=  len(v[dep_prop])*[i] # index of the study 
        i = i+1
    return np.array(Tt), np.array(At), np.array(Et) * error, np.array(It)

#def format_experimental_data(data : dict):
#    return D

'''
Wrapper functions: Temperature as independent variable
'''
def feval_S(param, T, D):
    return seebeck_s(param[0], D['l'])
    
def feval_kL(param, T, D):
    return kL_from_eta(param[-4:], param[0], D['kL_in'], T)
    
def feval_zT(param, T, D):
    return zT_from_eta(param, D['kL_in'], T, D['l'])


'''
Wrapper functions: Seebeck as independent variable. Proxy for doping level. Seebeck provided in muV/K
Returns carrier concnetration in cm^-3
'''

def feval_carrconc_S(param, S, D):
    S = S * 1e-6
    return carr_conc_from_Seebeck(param[0], S, D['T'], D['l'])

def feval_conductivity_S(param, S, D):
    S = S * 1e-6
    return conductivity_from_Seebeck(param[0], S, D['T'], D['l'])


'''
Sum together the likelihoods from different parameters

Currently, fitting to Seebeck, lattice thermal conductivity separately, and zT?
And then zT.
'''
def likelihood(param, D):
    dA_S = D['At_S'] - feval_S(param, D['Tt'], D)
    dA_kL = D['At_kL'] - feval_kL(param, D['Tt'], D)
    dA_zT = D['At_zT'] - feval_zT(param, D['Tt'], D)
#    dA_zT = D['At_zT'] - feval_zT_eta(param, D['Tt_zT'], D)
    #Obtain hyperparameters for the zT data: this might be for scaling?
    probS = ss.norm.logpdf(dA_S, loc=0, scale = D['Et_S']).sum() #need to actually add a parameter for the likelihood scale
    probkL = ss.norm.logpdf(dA_kL, loc=0, scale = D['Et_kL']).sum()
    probzT = ss.norm.logpdf(dA_zT, loc=0, scale = D['Et_zT']).sum()
    prob = probS + probkL + probzT
    if np.isnan(prob):
        return -np.inf
    return prob

def pisarenko_likelihood(param, D):
    dA_nH = D['At_nh'] - feval_carrconc_S(param, D['St_nh'], D)
    probnH = ss.norm.logpdf(dA_nH, loc = 0, scale = D['Et_nh']).sum()
    if np.isnan(probnH):
        return -np.inf
    return probnH

def jonker_likelihood(param, D):
    dA_c = D['At_c'] - feval_conductivity_S(param, D['St_c'], D)
    prob_c = ss.norm.logpdf(dA_c, loc = 0, scale = D['Et_c']).sum()
    if np.isnan(prob_c):
        return -np.inf
    return prob_c

#def jonker_likelihood():
#    return 
    
if __name__ == '__main__':
    #Fit the eta values to each Seebeck coefficient 
            # for convenience, store all important variables in dictionary
        D = {}
        
        #Initalize Temperature and scattering exponent
        D['T'] = 300
        D['l'] = 1
        
        D['name_list'] = ['Goyal2019Bi',	'Liu2012Sb',	'Liu2014Bi',	'Yin2016Sb',	'Zhang2019Sb']

        # outname is the name for plots, etc
        D['outname'] = '/Users/ramyagurunathan/Documents/PhDProjects/Argonne_TECCA/UQData/Mg2SiSn/outfiles/MgSi30Sn70_l1'

        # set up a log file
        D['wrt_file'] = D['outname'] + '.txt'
        fil = open(D['wrt_file'], 'w')
        fil.close()
        #Read experimental data into dictionary
        data = \
        read_data_pd('/Users/ramyagurunathan/Documents/PhDProjects/Argonne_TECCA/UQData/Mg2SiSn/xSi30', D['name_list'])
        
        #Get carrier concnetration from hall coefficient
        for k,v in data.items():
            v['nH (cm^-3)'] = list(1 / (np.array(v['RH (cm^3/C)']) * hpr.e))
        
#        #Seebeck data
#        D['Tt_S'], D['At_S'], D['Et_S'], D['It_S'],  = get_data(data, ind_prop = '', dep_prop = 'S (V/K)', 0.15)        
#        
        #Hall factor data
        D['St_nh'], D['At_nh'], D['Et_nh'], D['It_nh'] = get_data(data, ind_prop = 'S (muV/K)', dep_prop = 'nH (cm^-3)', error = 0.1)
        
        #Fit eta to the Seebeck value?
        #
        D['likelihood'] = pisarenko_likelihood
    
#        
        D['sampler'] = 'emcee'
#        
#        #Define prior distribution:
        D['distV'] = 1 * ['uniform']
#        
        D['dim'] = len(D['distV'])

        '''
        Define priors. Initalize property traces and trace plots
        '''
        D['pname'] = ['mstar']
        D['pname_plt'] = ['m^*']
        
        D['n_param'] = 1
        
        prior_file = '/Users/ramyagurunathan/Documents/PhDProjects/Argonne_TECCA/UQData/Mg2SiSn/outfiles/MgSi30Sn70_l1_param.csv'
        
        lb, ub = cc.load_next_prior(prior_file)
        if prior_file:
            D['locV'] = lb
            D['scaleV'] = ub - lb
        else:
            D['locV'] =  [0]
            D['scaleV'] = [5]
        
        
        sampler_dict = {'nlinks' : 300, 'nwalkers' : 10, 'ntemps' : 5, 'ntune' : 100}
        
        
        '''
        run MCMC and sample posterior
        '''
        D = cc.sampler_multip_emcee(D, sampler_dict)
            
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
        
        '''
        Plot Prediction
        '''
        cp.plot_prediction(flattrace, D['name_list'],
                   D['St_nh'], D['At_nh'], D['It_nh'], feval_carrconc_S, D,
                   colorL=['k', 'r', 'g', 'c', 'gold', 'darkorchid'], xlabel = 'S', ylabel = 'nH')
        
