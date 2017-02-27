#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:02:53 2017

@author: Hal
"""

import pandas as pd
from pandas import Series, DataFrame
from datetime import datetime
import numpy as np
#from scipy import signal
#from sklearn.decomposition import PCA
#import seaborn as sns
#from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.fft import fft, rfft
from random import random

def import_data(liste_fichiers, dossier):
    """
    importe les datas de tous les fichiers de la liste de noms de fichiers
    passée en argument.
    Pour chacun, le fichier ne doit pas comporter de valeurs manquantes (NA).
    L'en-tête doit être : 'date', 'Respiration', 'GSR', 'Temperature', 'CFM'.
    Séparateur de colonne : ';'
    Séparateur décimal : ','
    Renvoie un dictionnaire  de dataframes.
    """
    df_dict = {}
    dtype = {'Respiration':np.float64, 'GSR':np.float64, 'Temperature':np.float64, 'CFM':np.float64}
    
    n = len(liste_fichiers)
    
    for i, nom_fichier in enumerate(liste_fichiers):
        data = pd.read_csv('../data/' + dossier + '/' + nom_fichier, sep = ';', decimal =',', dtype = dtype)
        
        dates = []
        for date in data['date']:
            dates.append(datetime.strptime(date, '%d/%m/%Y %H:%M:%S.%f'))

        t = []
        for date in dates:
            t.append((date-dates[0]).total_seconds())
        
        time = Series(t, name='t')

        X = DataFrame(data[['Respiration', 'GSR', 'Temperature', 'CFM']], columns = ['t', 'Respiration', 'GSR', 'Temperature', 'CFM'])
        X['t'] = time

        df_dict[nom_fichier[:-4]] = X
        
        print("Fait pour", i+1, "fichiers sur", n, ".")
    
    return df_dict

def import_df(liste_fichiers, dossier):
    """
    Importe sous forme d'un dictionnaire de dataframes les données des fichiers
    csv dont la liste est passée en argument.
    """
    df_dict = {}
    dtype = {'t':np.float64, 'Respiration':np.float64, 'GSR':np.float64, 'Temperature':np.float64, 'CFM':np.float64}
    for i, nom_fichier in enumerate(liste_fichiers):
        data = pd.read_csv('../data/' + dossier + '/' + nom_fichier, sep = ';', decimal ='.', dtype = dtype)
        
        df_dict[nom_fichier[:-4]] = data
               
    return df_dict

def plot_exp(df, exp_name):
    """
    A partir d'une dataframe, fait un plot pour
    chaque variable.
    exp_name est le nom de l'expérience.
    """
    respi = df['Respiration']
    GSR = df['GSR']
    temp = df['Temperature']
    CFM = df['CFM']
    t = df['t']
        
    fig = plt.figure(figsize=(20, 10))
    
    ax_respi = fig.add_subplot(221)
    ax_GSR = fig.add_subplot(222)
    ax_temp = fig.add_subplot(223)
    ax_CFM = fig.add_subplot(224)
        
    ax_respi.set_title("Respiration")
    ax_GSR.set_title("GSR")
    ax_temp.set_title("Température")
    ax_CFM.set_title("CFM")
        
    ax_respi.plot(t, respi, color = "Blue", alpha = 0.7)
    ax_GSR.plot(t, GSR, color = "Green", alpha = 0.7)
    ax_temp.plot(t, temp, color = "Red", alpha = 0.7)
    ax_CFM.plot(t, CFM, color = "Grey", alpha = 0.7)
    
    plt.suptitle(exp_name, fontsize = 40)
    
def sinus(x):
    if np.sin(x) < 0:
        return 0
    else:
        return np.sin(x) + random()
    
def detect_var(t, X):
    """
    A partir des séries t et X (de même taille), renvoie :
    - une nouvelle série X2 où chaque série de valeurs redondantes à été remplacée par une
      unique valeur.
    - une liste t2 des milieux des intervalles de dates de chaque série de valeurs redondantes.
    """
    X2 = [X[0]]
    t2 = [t[0]]
    for i in range(1, len(X)): 
        if X[i] != X[i-1]:
            X2.append(X[i])
            t2[-1] = (t2[-1] + t[i-1])/2
            t2.append(t[i])
            
    return t2, X2

def affiche_fft(t, y, Ymax = None, fmax = None):
    """
    Affiche y(t) et sa FFT (Y)
    """
    n = len(t) # nb d'échantillons
    k = np.arange(0, n, 1) #arange(min, max, pas)
    f = k/(t[-1]-t[0]) # les fréquences
    Y = fft(y)/n # fft computing and normalization
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, y)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    
    ax[1].plot(f[:n//2], abs(Y[:n//2]), 'r') # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    
    if Ymax is not None:
        ax[1].set_ylim(0, Ymax)
    if fmax is not None:
        ax[1].set_xlim(0, fmax)
        
    return f[:n//2], abs(Y[:n//2])

def extract(df, var, debut, fin):
    """
    Renvoie df['t'] et df[var] pour t compris dans [debut:fin]
    """
    mask = (df['t'] > debut) & (df['t'] < fin)
    return list(df['t'][mask]), list(df[var][mask])

def freq(f, Y, debut):
    """
    Renvoie la fréquence correspond au max du spectre.
    debut est l'indice à partir duquel on cherche.
    """
    maxi = Y[debut]
    indice = debut
    for i in range(debut, len(Y)):
        if Y[i] > maxi:
            maxi = Y[i]
            indice = i
    return f[indice]

def calcul_freq(df, fen = 50):
    """
    Calcule la fréquence respiratoire f(t) à partir du signal
    y(t) de l'amplitude de déformation de la cage thoracique.
    fen est la fenêtre en secondes (50s = 10 périodes)
    df est la dataframe de l'expérience, qui contient toute les variables.
    """
    frequences = []
    dates = []
    tf = int(df['t'][len(df['t'])-1])
    for ti in range(fen//2, tf - fen//2):
        t, y = extract(df, 'Respiration', ti - fen//2, ti + fen//2)
    
        n = len(t) # nb d'échantillons
        k = np.arange(0, n, 1) #arange(min, max, pas)
        f = k/(t[-1]-t[0]) # les fréquences
        Y = fft(y)/n # fft computing and normalization
    
        frequences.append(freq(f[:n//2], abs(Y[:n//2]), 5))
        dates.append(ti)
    
    return dates, frequences
    
    
    
    
    
    