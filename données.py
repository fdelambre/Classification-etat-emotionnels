#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:10:54 2017

@author: Hal
"""

import csv

with open('AB_SurEchantillonHF.csv', 'r') as f:
    reader = csv.reader(f)

    for row in reader:
        print(', '.join(row))    
