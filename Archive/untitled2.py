# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 03:32:34 2025

@author: zarkm
"""
from Base import *
import numpy as np
from scipy.special import j0
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

# --- bounds & bookkeeping for coeff vector ---
nz = len(coeffs_z_norm)
nf = len(coeffs_f_norm)
nr = len(coeffs_rbf_norm)
# pack/unpack helpers
def pack(cz, cf, cr):    return np.concatenate([cz, cf, cr])
def unpack(x):
    return x[:nz], x[nz:nz+nf], x[nz+nf:]

# --- precompute static grids & target once ---
R       = 4e-3
Nr      = 200
r       = np.linspace(0, R, Nr)
z       = np.linspace(0.03, 0.21, 150)
r_cut_i = np.where(r <= 0.1e-3)[0]
lambda0 = 633e-9
k       = 2 * np.pi / lambda0

# desired target (area-matched later inside objective)
z0    = 0.07
sigma = 1e-3
I_des_raw = np.exp(-0.5 * ((z - z0)/sigma)**2)

# smoothing parameters
win_sg = 11; ord_sg = 3
sigma_g = 1.0

def simulate_curve(cz, cf, cr):
    """Given normalized coeff arrays, return I_smooth (length Nz)."""
    # 1) build h(r) via your Base.py function (already imported)
    h = abstract_profile_normalized(r/R, cz, cf, cr, 1.0)
    phi = 2 * np.pi * h

    # 2) propagate & fill I[iz,ir]
    Nz = len(z)
    I  = np.zeros((Nz, Nr))
    for iz, zi in enumerate(z):
        prefac = np.exp(1j * k * zi) / (1j * lambda0 * zi)
        Q      = np.exp(1j * k * r**2 / (2 * zi))
        for ir, ri in enumerate(r):
            kernel    = j0(k * ri * r / zi)
            integrand = np.exp(1j * phi) * Q * kernel * r
            E_r       = prefac * np.trapz(integrand, r)
            I[iz,ir]  = np.abs(E_r)**2

    # 3) integrate over r ≤ 0.1 mm
    I_int = np.trapz(I[:, :r_cut_i[-1]+1], x=r[:r_cut_i[-1]+1], axis=1)

    # 4) smooth (you may choose one)
    I_sg = savgol_filter(I_int, win_sg, ord_sg)
    I_sm = gaussian_filter1d(I_int, sigma=sigma_g)

    # choose one—for instance Gaussian
    return I_sm

def objective(x):
    """MSE between simulated curve and area-matched target."""
    cz, cf, cr = unpack(x)
    I_sim = simulate_curve(cz, cf, cr)

    # area-match target
    area_sim = np.trapz(I_sim, z)
    area_des = np.trapz(I_des_raw, z)
    I_tar    = I_des_raw * (area_sim/area_des)

    # mean squared error
    return np.mean((I_sim - I_tar)**2)

# --- run the optimizer ---
x0 = pack(coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm)
bounds = [(0,1)] * (nz+nf+nr)

res = minimize(objective, x0,
               method='L-BFGS-B',
               bounds=bounds,
               options={'maxiter':50, 'ftol':1e-6, 'disp':True})

# --- unpack optimized coeffs and show result ---
cz_opt, cf_opt, cr_opt = unpack(res.x)
I_opt = simulate_curve(cz_opt, cf_opt, cr_opt)

# --- plot final comparison ---
import matplotlib.pyplot as plt
area_sim = np.trapz(I_opt, z)
area_des = np.trapz(I_des_raw, z)
I_tar   = I_des_raw * (area_sim/area_des)

plt.figure(figsize=(8,4))
plt.plot(z, I_opt,  label='Optimized (smoothed)')
plt.plot(z, I_tar,  '--', label='Desired Gaussian')
plt.xlabel('Propagation distance $z$ (m)')
plt.ylabel('Integrated intensity')
plt.title('Optimized Through-Focus vs. Target')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
