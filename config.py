# -*- coding: utf-8 -*-
"""
Created on Sun May  3 02:16:49 2026

@author: zarkm
"""
import numpy as np

# ------
R         = 4e-3
Nr        = 1000 #1000
r         = np.linspace(0, R, Nr)
z         = np.linspace(0.03, 0.21, 150)
lambda0   = 633e-9
lambda1   = 445e-9
lambda2   = 532e-9

z0, z1, z2, sigma, sigma1, sigma2 = 0.07, 0.11, 0.17, 1e-3, 1e-3, 1e-3


n_zernike = 6
n_fourier = 350
n_rbf = 7