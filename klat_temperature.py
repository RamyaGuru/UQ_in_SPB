#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 22:09:50 2020

@author: ramyagurunathan

High Temperature (above Debye T Thermal Model)
"""

import numpy as np
from math import pi, exp, sin
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

np.seterr(divide='raise', invalid="raise")

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
    A = (((6 * pi**2)**(2/3) / (4 * pi**2)) * param[0] * (avgM / hpr.Na) *\
         1e-3  * vs**3 / (param[1]**2 * avgV**(2/3))) * (N**(-1/3))
    B = param[2] * (3  * hpr.kB * vs / (2 * avgV**(2/3)))  * (pi / 6)**(1/3) *\
    (1 - N**(-2/3))
    return A * (1 / T) + B



'''
Phonon Dispersion Approximations
'''   

def debye_model_freq(atmV, vs, k):
    return vs * k

def bvk_model_freq(atmV, vs, k):
    kmax = (6 * pi**2 / atmV)**(1/3)
    return vs*(2/pi)*(kmax/k)*sin((pi/2)*(k/kmax)) * k

#Add polynoimal dispersion from Schrade 2018


'''
Relaxation time expressions
'''

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

def pd_tau_k(A, gamma, k, avgV, vs, disp_fxn):
    freq = disp_fxn(avgV, vs, k)
    return A * 4 * pi * vs/ (avgV * freq**2 * k**2 * gamma)

def umklapp_tau_k(B, grun2, k, avgM, avgV, vs, disp_fxn, T):
    return B * (6 * pi**2)**(1/3)/2 * ((avgM / hpr.Na) * 1e-3 * vs)\
 / (hpr.kB * avgV**(1/3) * float(grun2) * k**2 * T) #grun2 is the squared gruneisen parameter
 
def normal_TA_tau_k(B1, freq, grun2, avgM, avgV, vs, T): #look into prefactor
    return B1 * freq * T
 
def boundary_tau(C, vs, d):
    '''
    Include additional forms of boundary scattering? d_s + d_c?
    '''
    return C * (d / vs)



'''
Thermal Conductivity integral over k
'''
def spectral_C_k(avgV, vs, k, disp_fxn, T): 
    freq = disp_fxn(avgV, vs, k)
    x = hpr.hbar * freq / (hpr.kB * T)
    return (1 / (2 * pi**2)) * hpr.kB * k**2 * (x**2 * np.exp(x))/ (np.exp(x) - 1)**2

def kL_umklapp_PD_vs_T_k(param, avgV, N, vs, stoich, og, subst, nk, c, disp_fxn, T):
    '''
    param = [coeff1, coeff2, gruneisen, epsilon]
    '''
    gammaM = gamma(stoich, og['mass'], subst['mass'], c)
    gammaV = gamma(stoich, og['rad'], subst['rad'], c)
    gammatot = gammaM + param[3] * gammaV
    
    avgM = sum((1-c) * np.array(og['mass']) + c * np.array(subst['mass'])) / sum(stoich)
    kL = 0
    kmax = (6 * pi**2 / avgV)**(1/3)
    dk = kmax / nk
    for k in np.arange(dk, kmax, dk):
        tauPD = pd_tau_k(param[0], gammatot, k, avgV, vs, disp_fxn)
        tauU = umklapp_tau_k(param[1], param[2], k, avgM, avgV, vs, disp_fxn, T)
        tau = 1/(tauU**(-1) + tauPD**(-1))
        kL = kL + spectral_C_k(avgV, vs, k, disp_fxn, T) * vs**2 * tau * dk
    return kL

def kL_umklapp_PD_b_vs_T_k(param, avgV, N, vs, d, stoich, og, subst, nk, c, disp_fxn, T):
    '''
    param = [coeff1, coeff2, gruneisen, epsilon]
    '''
    gammaM = gamma(stoich, og['mass'], subst['mass'], c)
    gammaV = gamma(stoich, og['rad'], subst['rad'], c)
    gammatot = gammaM + param[3] * gammaV
    
    avgM = sum((1-c) * np.array(og['mass']) + c * np.array(subst['mass'])) / sum(stoich)
    kL = 0
    kmax = (6 * pi**2 / avgV)**(1/3)
    dk = kmax / nk
    for k in np.arange(dk, kmax, dk):
        tauPD = pd_tau_k(param[0], gammatot, k, avgV, vs, disp_fxn)
        tauU = umklapp_tau_k(param[1], param[2], k, avgM, avgV, vs, disp_fxn, T)
        tauB = boundary_tau(param[4], vs, d)
        tau = 1/(tauU**(-1) + tauPD**(-1) + tauB**(-1))
        kL = kL + spectral_C_k(avgV, vs, k, disp_fxn, T) * vs**2 * tau * dk
    return kL

def kL_ac_opt(param, avgV, N, vs, stoich, og, subst, nk, c, disp_fxn, kL_ac_fxn, T):
    kL_ac = kL_ac_fxn(param[:-1], avgV, N, vs, stoich, og, subst, nk, c, disp_fxn, T) #Integrate up to small BZ k_max for BvK dispersion
    kL_opt = param[-1] * hpr.kB * vs / (avgV**(2/3)) * (pi / 6)**(1/3) 
    full_kL = kL_ac * N**(-1/3) + kL_opt * (1 - N**(-2/3))
    return full_kL
'''
Wrapper functions
'''

def feval_klat(param, T, D):
    kL = kL_T(param, D['avgM'], D['avgV'], D['N'], D['vs'],T)
    return kL


def feval_klat_PD_U_b(param, T, D):
    kL = kL_umklapp_PD_b_vs_T_k(param, D['avgV'], D['N'], D['vs'], D['d'], D['stoich'],\
                            D['og'], D['subst'], D['nfreq'], D['c'], T)
    return kL 

def feval_klat_PD_U_k(param, T, D):
    kL = kL_umklapp_PD_vs_T_k(param, D['avgV'], D['N'], D['vs'],  D['stoich'],\
                            D['og'], D['subst'], D['nk'], D['c'], D['disp_fxn'], T)
    return kL

def feval_klat_ac_opt(param, T, D):
    kL = kL_ac_opt(param, D['avgV'], D['N'], D['vs'],  D['stoich'],\
                            D['og'], D['subst'], D['nk'], D['c'], D['disp_fxn'],\
                            D['kL_ac_fxn'], T)
    return kL

#Can this just be added to the core_compute module?
def likelihood(param, D):
    feval_fxn = D['feval_klat_fxn']
    model_val = feval_fxn(param, D['Tt'], D)
    dA = D['At'] - model_val
    #Obtain hyperparameters for the zT data: this might be for scaling?
    #Try alternative dsitirubtions fro the log likelihood
    prob = ss.norm.logpdf(dA, loc=0, scale = D['Et']).sum()
    if np.isnan(prob):
        return -np.inf    
    return prob    


def read_data(data_dir,npts, dopant_conc : str):
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
    Et = np.zeros(At.shape) #experimental error bars? initialize these values and use in likelihood
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
    Et = []
    try:
        for s in subdirs:
            os.chdir(s)
            for file in os.listdir(os.getcwd()):
                if fnmatch.fnmatch(file, '*x' + dopant_conc + '.txt'):
                    df = pd.read_csv(file, delimiter = '\t')
                    data = df.to_dict('list')
                    Tt += list(df['T(K)'])
                    At += data['kL (W/m/K)']
                    Et += data['20%eps kL (W/m/K)']
#                    Et += [d/4 for d in data['20%eps kL (W/m/K)']] #is this too huge?
                else:
                    continue
    except:
        raise hpr.PropertyError('Data folders not formatted properly')
    Tt = np.array(Tt)
    At = np.array(At)
    Et = np.array(Et)
    It = np.zeros(At.shape)           
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
        D['distV'] = 5 * ['uniform']
        prior_file = '/Users/ramyagurunathan/Documents/PhDProjects/Argonne_TECCA/UQData/FeNbSb/Titanium/Fu2014EES/' + D['outname'] + '_param.csv'
        if prior_file:
            lb, ub = cc.load_next_prior(prior_file)
            D['scaleV'] = [u - l for u,l in zip(ub, lb)]
            D['locV'] = list(lb)
        else:
            D['scaleV'] = [3,3, 3, 600, 2]
            D['locV'] = [0,0,0,0,0]

        D['dim'] = len(D['distV'])
        
        D['pname'] = ['A', 'B', 'gruneisen', 'epsilon', 'C']
        D['pname_plt'] = ['A', 'B', r'\gamma', r'\epsilon', 'C'] 
        
        D['n_param'] = 5
        
        '''
        Set material constants
        '''
#        Fe2VAl
#        D['vs'] = v_sound(7750, 4530)
#        D['avgM'] = 47.403
#        D['avgV'] = 11.6e-30 #in cubic meters
#        D['N'] = 4
        '''
        FeNbSb
        '''
        D['stoich'] = [1,1,1]
        D['nk'] = 100
        D['og'] = {'mass': [55.845, 92.906, 121.76],'rad': [.75, .86, 0.9]}
        D['subst'] = {'mass': [55.845, 47.867, 121.76] ,'rad': [.75, .67, 0.9]}
        D['c'] = .2
        D['vs'] = 3052
        D['avgV'] = (53.167E-30) / 3
        D['avgM'] = 90.17
        D['N'] = 3
        D['d'] = 350E-9
        D['disp_fxn'] = debye_model_freq
        D['kL_ac_fxn'] =  kL_umklapp_PD_vs_T_k
        D['feval_klat_fxn'] = feval_klat_ac_opt
        sampler_dict = {'nlinks' : 300, 'nwalkers' :10, 'ntemps' : 5, 'ntune' : 100}
        
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
                           D['Tt'], D['At'], D['It'], feval_klat_PD_U_k, D,
                           colorL=['k'], xlabel = 'T (K)', ylabel = r'$\kappa_L$ (W/m/K)')
            