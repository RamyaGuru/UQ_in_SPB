#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 23:04:53 2020

@author: ramyagurunathan

Helper Script for the SPB UQ Model
"""

'''
Constants
'''

kB = 1.38e-23 # V / K
h = 6.626e-34 # J * s
hbar = 1.054e-34 # J * s

e = 1.602e-19 # C
me = 9.11e-31 # kg
Na = 6.02e23 # /mol

'''
User-defined exceptions
'''
class PropertyError(Exception):
    def __init__(self, message):
        self.message = message

