# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:40:43 2026

@author: zarkm
"""

from basis import abstract_profile_normalized
import numpy as np
from scipy.special import j0
from scipy.ndimage import gaussian_filter1d

def simulate_onaxis(cz, cf, cr, r, z, R, lambdaa):
    """
    Compute only I(z, r=0) for each z (on-axis intensity).
    """
    # 1) radial profile and phase
    h   = abstract_profile_normalized(r/R, cz, cf, cr, 1.0)
    phi = 2*np.pi * h * (633e-9 / lambdaa)              # shape (Nr,) #added lambda dependent phase scaling

    k = 2*np.pi / lambdaa
    Nz = len(z)

    I_onaxis = np.zeros(Nz)
    for iz, zi in enumerate(z):
        prefac = np.exp(1j*k*zi) / (1j * lambdaa * zi)
        Q      = np.exp(1j * k * r**2 / (2*zi))
        integrand = np.exp(1j*phi) * Q * r
        E0 = prefac * np.trapezoid(integrand, r)
        I_onaxis[iz] = np.abs(E0)**2
    #I_smoothed = gaussian_filter1d(I_onaxis, sigma=0.5)

    return I_onaxis

def simulate_curve(cz, cf, cr, r, z, R, lambda0):
    # radial profile h(r)
    h   = abstract_profile_normalized(r/R, cz, cf, cr, 1.0)
    phi = 2*np.pi * h

    k = 2*np.pi / lambda0
    Nz, Nr = len(z), len(r)

    # compute I[iz,ir]
    I = np.zeros((Nz, Nr))
    for iz, zi in enumerate(z):
        prefac = np.exp(1j*k*zi) / (1j*lambda0*zi)
        Q      = np.exp(1j * k * r**2 / (2*zi))
        for ir, ri in enumerate(r):
            kernel    = j0(k * ri * r / zi)
            integrand = np.exp(1j*phi) * Q * kernel * r
            E_r       = prefac * np.trapezoid(integrand, r)
            I[iz,ir]  = np.abs(E_r)**2

    # integrate over r ≤ 0.1 mm
    idx_cut = np.where(r <= 0.1e-3)[0]
    I_int = np.trapezoid(I[:, :idx_cut[-1]+1],
                    x=r[:idx_cut[-1]+1], axis=1)

    # smooth in z (choose one)
    #I_sg = savgol_filter(I_int, 11, 3)
    I_sm = gaussian_filter1d(I_int, sigma=1.0)

    return I_sm
