#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:22:16 2020

@author: ramyagurunathan
lattice thermal conductivity versus temperature
"""

import klat_temperature as kt
import os
import fnmatch
import numpy as np
import pandas as pd
import re
import helper as hpr
import UnaryBayes.core_compute as cc
import UnaryBayes.core_plot as cp
import pickle
import sys
from math import pi


def read_comp_data(data_dir, T = 300):
    '''
    Output dictionary with the structure:
        data['composition'] = np.array([[Temperatures], [kappa values]])
    '''
    os.chdir(str(data_dir))
    subdirs = [x[0] for x in os.walk(data_dir)]
    Tt = []
    At = []
    Et = []
    It = []
    regex = re.compile(r'\d+')
    try:
        i = 1
        for s in subdirs:
            os.chdir(s)
            for file in os.listdir(os.getcwd()):
                if fnmatch.fnmatch(file, '*x*.txt'):
                    comp = regex.findall(file)
                    c = int(comp[0]) * 1E-2
                    Tt.append(c)
                    df = pd.read_csv(file, delimiter = '\t')
                    At.append(df[df['T(K)'] == T]['kL (W/m/K)'][0])
                    Et.append(df.loc[df['T(K)'] == T]['20%eps kL (W/m/K)'][0])
                    It.append(i)
            i = i+1
    except:
        raise hpr.PropertyError('Data folders not formatted properly')
    Tt = np.array(Tt)
    At = np.array(At)
    Et = np.array(Et) 
    It = np.zeros(At.shape)         
    return Tt, At, Et, It
            
                    
                    
def kL_alloy(param, atmV, vs, stoich, og, subst, kap_pure, c):
    gammaM = kt.gamma(stoich, og['mass'], subst['mass'], c)
    gammaV = kt.gamma(stoich, og['rad'], subst['rad'], c)
    gammatot = gammaM + param[0] * gammaV
    prefix = (6**(1/3)/2)*(pi**(5/3)/hpr.kB)*(atmV**(2/3)/vs)
    u = (prefix*gammatot*kap_pure)**(1/2)
    kL = kap_pure*np.arctan(u)/u
    return kL
    
'''
Wrapper functions
'''              
def feval_klat_alloy(param, c, D):
    kL = kL_alloy(param, D['avgV'], D['vs'], D['stoich'], D['og'], D['subst'], D['kap_pure'], c)
    return kL  


if __name__ == '__main__':
    # for convenience, store all important variables in dictionary
    D = {}
    
    # save the current file name
    D['fname'] = sys.argv[0]
    
    # outname is the name for plots, etc
    D['outname'] = 'Ti_FeNbSb_klat_alloy'
    
    # set up a log file
    D['wrt_file'] = D['outname'] + '.txt'
    fil = open(D['wrt_file'], 'w')
    fil.close()
    #        D['Tt'], D['At'], D['Et'], D['It'] = \
    #        read_data('/Users/ramyagurunathan/Documents/PhDProjects/Fe2VAl/Fe2VAl/Data/Cobalt', 6, '20')
    D['Tt'], D['At'], D['Et'], D['It'] = \
    read_comp_data('/Users/ramyagurunathan/Documents/PhDProjects/Argonne_TECCA/UQData/FeNbSb/Titanium',T = 300) 
    D['likelihood'] = kt.likelihood
    
    D['name_list'] = ['Ti_in_FeNbSb']
    
    D['sampler'] = 'emcee'
    
    '''
    Define prior distributions. Do I need an 'epsilon' factor?
    '''
    D['distV'] = 1 * ['uniform']
    D['scaleV'] = [10]
    D['locV'] = [25] #centers of distributions
    
    #D['scaleV'] = [1.65 - 0.95, 1.16 - 0.60, 1.68 - 1.07, 324 - 157]
    #D['locV'] = [0.95, 0.60, 1.07, 157]
    
    D['dim'] = len(D['distV'])
    
    D['pname'] = ['epsilon']
    D['pname_plt'] = [r'\epsilon'] 
    
    D['n_param'] = 1
    
    '''
    FeNbSb
    '''
    D['stoich'] = [1,1,1]
    D['og'] = {'mass': [55.845, 92.906, 121.76],'rad': [.75, .86, 0.9]}
    D['subst'] = {'mass': [55.845, 47.867, 121.76] ,'rad': [.75, .67, 0.9]}
    D['vs'] = 3052
    D['avgV'] = (53.167E-30) / 3
    D['kap_pure'] = 17
#    D['avgM'] = 90.17
#    D['N'] = 3
#    D['d'] = 350E-9
#    D['disp_fxn'] = kt.debye_model_freq 
#    D['kL_ac_fxn'] =  kt.kL_umklapp_PD_vs_T_k
    D['feval_klat_fxn'] = feval_klat_alloy
    sampler_dict = {'nlinks' : 300, 'nwalkers' :10, 'ntemps' : 1, 'ntune' : 100}
    
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
    
    #    
    #    
    cp.plot_prediction(flattrace, D['name_list'],
                       D['Tt'], D['At'], D['It'], feval_klat_alloy, D,
                       colorL=['k', 'r'], pltname = D['outname'], xlim = [0.0001, 0.999], xlabel = 'x', ylabel = r'$\kappa_L$ (W/m/K)')