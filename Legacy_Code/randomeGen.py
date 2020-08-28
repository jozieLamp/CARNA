'''
HemoPheno4HF
SCRIPT DESCRIPTION: Random Number Generator for Parameters
CODE DEVELOPED BY: Yuxin Wu
ORGANIZATION: University of Virginia, Charlottesville, VA
LAST UPDATED: 8/24/2020
'''

import random
import numpy as np
# RAP=[]
# PAS=[]
# PAD=[]
# PAMN=[]
# PCWP=[]
# PCWPMod=[]
# PCWPA=[]
# PCWPMN=[]
# CO=[]
# CI=[]
# SVRHemo=[]
# MIXED=[]
# BPSYS=[]
# BPDIAS=[]
# HRTRT=[]
# RATHemo=[]
# MAP=[]
# MPAP=[]
# CPI=[]
# PP=[]
# PPP=[]
# PAPP=[]
# SVR=[]
# RAT=[]
# PPRatio=[]
# # Age=[]
# EjF=[]
hemoDict = {'RAP': [0.0, 85.0],
            'PAS': [0.0, 90.0],
            'PAD': [0.0, 59.0],
            'PAMN': [0.0, 82.0],
            'PCWP': [0.0, 53.0],
            'PCWPMod': [0.0, 53.0],
            'PCWPA': [0.0, 53.0],
            'PCWPMN': [0.0, 49.0],
            'CO': [0.0, 25.0],
            'CI': [0.0, 4.81],
            'SVRHemo': [0.0, 6866.0],
            'MIXED': [0.0, 627.0],
            'BPSYS': [0.0, 168.0],
            'BPDIAS': [0.0, 125.0],
            'HRTRT': [0.0, 125.0],
            'RATHemo': [0.0, 3.74],
            'MAP': [0.0, 226.0],
            'MPAP': [0.0, 129.3333333],
            'CPI': [0.0, 1.820192166],
            'PP': [-55.0, 106.0],
            'PPP': [-0.833333333, 0.736842105],
            'PAPP': [0.0, 0.8444444440000001],
            'SVR': [0.0, 10755.555559999999],
            'RAT': [0.0, 3.736842105],
            'PPRatio': [-0.8870967740000001, 9.666666667000001],
            'Age': [23.0, 88.0],
            'EjF': [0.0, 45.0]}
a=[]
for i in hemoDict.keys():
    up = round(hemoDict.get(i)[0])
    low = round(hemoDict.get(i)[1])
    key= i
    a=[]
    for j in range(0,6):
        # r = (up - low)/10

        a.append(random.uniform(low,up))
    print("'"+i+"'",": ",a,",")
