#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:31:28 2020

@author: ramyagurunathan

Jonker scattering mechanism test
"""

import spb_scattering_mech as sm
import UnaryBayes.core_compute as cc
import UnaryBayes.core_plot as cp
import numpy as np
import helper as hpr
import pickle
import sys

D = {}

#Initalize Temperature and scattering exponent
D['T'] = 300
D['l'] = 0.5

D['name_list'] = ['Goyal2019Bi',	'Liu2012Sb',	'Liu2014Bi',	'Yin2016Sb',	'Zhang2019Sb']

# outname is the name for plots, etc
D['outname'] = '/Users/ramyagurunathan/Documents/PhDProjects/Argonne_TECCA/UQData/Mg2SiSn/outfiles/MgSi30Sn70'

# set up a log file
D['wrt_file'] = D['outname'] + '.txt'
fil = open(D['wrt_file'], 'w')
fil.close()
#Read experimental data into dictionary
data = \
sm.read_data_pd('/Users/ramyagurunathan/Documents/PhDProjects/Argonne_TECCA/UQData/Mg2SiSn/xSi30', D['name_list'])

#Get carrier concnetration from hall coefficient
for k,v in data.items():
    v['nH (cm^-3)'] = list(1 / (np.array(v['RH (cm^3/C)']) * hpr.e))

#        #Seebeck data
#        D['Tt_S'], D['At_S'], D['Et_S'], D['It_S'],  = get_data(data, ind_prop = '', dep_prop = 'S (V/K)', 0.15)        
#        
#Hall factor data
D['St_c'], D['At_c'], D['Et_c'], D['It_c'] = sm.get_data(data, ind_prop = 'S (muV/K)', dep_prop = 'sigma (S/cm)', error = 0.05)


#Fit eta to the Seebeck value?
#
D['likelihood'] = sm.jonker_likelihood

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
D['pname'] = ['weighted_mobility']
D['pname_plt'] = ['muW']

D['n_param'] = 1

D['locV'] =  [134e-4]
D['scaleV'] = [148.5e-4 - 134e-4]


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
           D['St_c'], D['At_c'], D['It_c'], sm.feval_conductivity_S , D,
           colorL=['k', 'r', 'g', 'c', 'gold', 'darkorchid'], xlabel = 'S', ylabel = 'sigma ')


