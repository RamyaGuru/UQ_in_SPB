#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:21:35 2020

@author: ramyagurunathan

Two Band Model
"""

import single_parabolic_band as spb

'''
Two band Model Functions

User specifies the relative curvature of the two bands (i.e. both electron, 
both hole, or one electorn and one hole pocket)
    --> could this just happen through the prior distribution assignment?
'''

def two_band_geom(band1 = 'electron', band2 = 'electron'):
    valid = ['electron', 'hole']
    if band1 not in valid or band2 not in valid:
        raise ValueError('band assignment must be electron or hole')
    return {'1': band1, '2' : band2}
    

def carrconc_from_eta(eta_1, eta_2, mstar1, mstar2, geom : dict):
    '''
    Carrier concentration in m^{-3}
    '''
    if geom['1'] == 'electron': sgn1 = 1
    else : sgn1 = -1
    if geom['2'] == 'electron': sgn2 = 1
    else : sgn2 = -1
    return abs(sgn1 * spb.carr_conc_from_eta(eta_1, mstar1) + sgn2 * spb.carr_conc_from_eta(eta_2, mstar2))

def twoband_conductivity(eta_n, eta_p, sigmae01, sigmae02, s=1):
    '''
    Note: sigmae01 should be the n-type value
    Conductivity in S/m
    '''
    return spb.conductivity(eta_n, sigmae01, s=1) + spb.conductivity(eta_p, sigmae02, s=1)

def twoband_seebeck(eta_1, eta_2, sigmae01, sigmae02, geom : dict):
    '''
    Note: sigmae01 should be the n-type value
    Seebeck in V/K
    '''
    if geom['1'] == 'electron': sgn1 = -1
    else : sgn1 = 1
    if geom['2'] == 'electron': sgn2 = -1
    else : sgn2 = 1
    return (sgn1 * spb.seebeck(eta_1) * spb.conductivity(eta_1, sigmae01, s=1) + sgn2 * spb.seebeck(eta_2) *\
 spb.conductivity(eta_2, sigmae02, s=1)) / (twoband_conductivity(eta_1, eta_2, sigmae01, sigmae02, s=1))

'''
Without the interband scattering
'''    
def twoband_lorenz(eta_1, eta_2, sigmae01, sigmae02, geom : dict):
    return spb.lorentz(eta_1) * spb.conductivity(eta_1, T,)

'''
With interband scattering
'''
def towband_lorenz_IB(eta_1, eta_2, sigmae01, sigmae02, geom : dict):
    
    return