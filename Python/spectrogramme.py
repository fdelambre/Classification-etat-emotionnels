#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:03:43 2017

@author: Hal
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, rfft
from scipy import signal
from math import sqrt

nb_spectres = 30
taille_window = 256
overlap = 32
N_ech = taille_window * nb_spectres - (nb_spectres-1) * overlap

# Calcul des ti, milieux de chaque fenêtre                                      
#time = [taille_window/2]                                      
#for i in range(nb_spectres-1):
#    time.append(time[-1]+taille_window-overlap)                                      
#print(time)

def f(t):
    """
    Fonction fréquence : elle varie avec le temps.
    """
    f0 = 1
    return f0 + 5*t 

def creation_signal():
    T = np.linspace(0, 10, N_ech)
    Y = []
    for t in T:
        Y.append(np.sin(2*np.pi*f(t)*t))
    return T, Y 

def affiche_signal(T, Y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(T, Y, color = 'blue', alpha = 0.8, linewidth = 2)

T, Y = creation_signal()
#affiche_signal(T, Y)
   
freq, time, Sxx = signal.spectrogram(Y)

def affiche_spectrogramme(freq, time, Sxx):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.pcolormesh(time, freq, Sxx, cmap='RdBu')
    #(fig.colorbar()
    ax.set_ylabel('Fréquence [Hz]')
    ax.set_xlabel('Temps [sec]')
    cbar = fig.colorbar(cax)

affiche_spectrogramme(freq, time, Sxx)
#plt.pcolormesh(time, freq, Sxx, cmap='RdBu')
print(len(freq), len(time), len(Sxx), len(Sxx[0]))
print('time')
print(time)

