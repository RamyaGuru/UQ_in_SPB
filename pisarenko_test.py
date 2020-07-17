#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:32:05 2020

@author: ramyagurunathan

Pisarenko Test
"""

def read_data_pd(data_dir):
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
                    Tt += list(df['n(K)'])
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
    Et = np.array(At * 0.15)          
    return Tt, At, Et, It
    

