#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:25:20 2020

@author: ramyagurunathan

Run Thermal Model
"""
import klat_temperature as kt
import UnaryBayes.core_compute as cc
import UnaryBayes.core_plot as cp
import pickle
import sys



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
kt.read_data_pd('/Users/ramyagurunathan/Documents/PhDProjects/Argonne_TECCA/UQData/FeNbSb/Titanium','20') 
D['likelihood'] = kt.likelihood

D['name_list'] = ['Ti in FeNbSb']

D['sampler'] = 'emcee'

'''
Define prior distributions. Do I need an 'epsilon' factor?
'''
D['distV'] = 4 * ['uniform']
D['scaleV'] = [2, 2, 2, 600]
D['locV'] = [0, 0, 0, 0] #centers of distributions
D['dim'] = len(D['distV'])

D['pname'] = ['A', 'B', 'gruneisen', 'epsilon']
D['pname_plt'] = ['A', 'B', r'\gamma', r'\epsilon'] 

D['n_param'] = 4

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
D['disp_fxn'] = kt.debye_model_freq 
sampler_dict = {'nlinks' : 200, 'nwalkers' :100, 'ntemps' : 1, 'ntune' : 200}

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
                   D['Tt'], D['At'], D['It'], kt.feval_klat_PD_U_k, D,
                   colorL=['k'], xlabel = 'T (K)', ylabel = r'$\kappa_L$ (W/m/K)')
    