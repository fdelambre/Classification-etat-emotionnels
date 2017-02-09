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


fe = 150.0  # fréquence d'échantillonnage
Te = 1/fe # période correspondante = pas de temps

tmin = 1.5
tmax = 2.3

# nb d'échantillons
n = np.rint((tmax-tmin)/Te)    
print(n)
    
t = np.arange(tmin, tmax, Te) # vecteur des dates
             
f = 5;   # une fréquence 
# le signal avec deux fréquences 5 et 10
y = np.sin(2*np.pi*f*t) + 2*np.sin(2*np.pi*(2*f)*t) 

n = len(y) # nb d'échantillons
print(n)
       
# vecteur des indices k       
k = np.arange(0, n, 1) #arange(min, max, pas)
freq = k/(tmax-tmin) # les fréquences

Y = fft(y)/n # fft computing and normalization

# Affichage
def affiche_fft(t, y, Y):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, y)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')

    ax[1].plot(freq[:n//2], abs(Y[:n//2]), 'r') # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')

affiche_fft(t, y, Y)
