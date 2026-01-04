# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 04:21:28 2025

@author: zarkm
"""

from Base import *
import numpy as np
from scipy.special import j0
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import numpy as np
from scipy.optimize import minimize
import pickle

# --- reuse your problem setup & bb_objective/unpack ---
R         = 4e-3
Nr        = 200
r         = np.linspace(0, R, Nr)
z         = np.linspace(0.03, 0.21, 150)
lambda0   = 633e-9
z0, sigma = 0.07, 1e-3
I_des_raw = np.exp(-0.5 * ((z - z0)/sigma)**2)

# Normalized random coefficients in [0,1]
coeffs_z_norm = np.random.rand(6)    # 6 Zernike terms
coeffs_f_norm = np.random.rand(60)   # 50 Fourier terms
coeffs_rbf_norm = np.random.rand(10) # 10 RBF terms


dims      = (len(coeffs_z_norm), len(coeffs_f_norm), len(coeffs_rbf_norm))
total_dim = sum(dims)

def unpack(x):
    Nz, Nf, Nrbf = dims
    return x[:Nz], x[Nz:Nz+Nf], x[Nz+Nf:]


def simulate_onaxis(cz, cf, cr, r, z, R, lambda0):
    """
    Compute only I(z, r=0) for each z (on-axis intensity).
    """
    # 1) radial profile and phase
    h   = abstract_profile_normalized(r/R, cz, cf, cr, 1.0)
    phi = 2*np.pi * h               # shape (Nr,)

    k = 2*np.pi / lambda0
    Nz = len(z)

    I_onaxis = np.zeros(Nz)
    for iz, zi in enumerate(z):
        prefac = np.exp(1j*k*zi) / (1j * lambda0 * zi)
        Q      = np.exp(1j * k * r**2 / (2*zi))
        integrand = np.exp(1j*phi) * Q * r
        E0 = prefac * np.trapz(integrand, r)
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
            E_r       = prefac * np.trapz(integrand, r)
            I[iz,ir]  = np.abs(E_r)**2

    # integrate over r ≤ 0.1 mm
    idx_cut = np.where(r <= 0.1e-3)[0]
    I_int = np.trapz(I[:, :idx_cut[-1]+1],
                    x=r[:idx_cut[-1]+1], axis=1)

    # smooth in z (choose one)
    #I_sg = savgol_filter(I_int, 11, 3)
    I_sm = gaussian_filter1d(I_int, sigma=1.0)

    return I_sm

def bb_objective(x):
    cz, cf, cr = unpack(x)
    I_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda0)
    area_sim = np.trapz(I_sim, z)
    area_des = np.trapz(I_des_raw, z)
    I_tar    = I_des_raw * (area_sim/area_des)
    return np.mean((I_sim - I_tar)**2)

# initial guess
x0 = np.concatenate([coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm])

# callback to print progress
best = {"val": bb_objective(x0)}
counter = {"it": 0}
def powell_callback(xk):
    counter["it"] += 1
    val = bb_objective(xk)
    if val < best["val"]:
        best["val"] = val
        print(f"Iter {counter['it']:4d}: New best MSE = {val:.6e}")
    return

# run Powell
res = minimize(
    bb_objective,
    x0,
    method='Nelder-Mead',
    callback=powell_callback,
    options={
        'maxiter': 4000000,
        'xtol': 1e-3,
        'ftol': 1e-3,
        'disp': True
    }
)

print("\nPattern search (Powell) finished.")
print("  Success:", res.success)
print(f"  Final best MSE = {res.fun:.6e}")

# unpack final
cz_opt, cf_opt, cr_opt = unpack(res.x)
I_opt = simulate_onaxis(cz_opt, cf_opt, cr_opt, r, z, R, lambda0)
# (You can now plot I_opt vs. the target as before.)


area_sim = np.trapz(I_opt, z)
area_des = np.trapz(I_des_raw, z)
I_fin    = I_tar * (area_sim/area_des)

# 4) plot
plt.figure(figsize=(8,4))
plt.plot(z, I_int,   label='Raw integrated')
#plt.plot(z, I_smooth, label='savgol Smoothed', linewidth=2)
plt.plot(z, I_opt, label='Gaussianly Smoothed', linewidth=2)
plt.plot(z, I_tar, label='Desired Focus (area-matched)', linestyle='--', linewidth=2)

plt.xlabel('Propagation distance $z$ (m)')
plt.ylabel('Integrated intensity $\int_0^{0.1\,\mathrm{mm}} I(r)\,dr$ (arb. units)')
plt.title('Through‐Focus Intensity Profile')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


h = abstract_profile_normalized(r, cz_opt, cf_opt, cr_opt, R)

plt.figure(figsize=(6, 4))
plt.plot(r, h, color='darkorange')
plt.xlabel("Normalized Radius r")
plt.ylabel("Height h(r)")
plt.title("Profile from Coeffs")
plt.grid(True)
plt.tight_layout()
plt.show()




# 1) Gather everything you want to save into a dict
vars_to_save = {
    'r'             : r,
    'z'             : z,
    'R'             : R,
    'lambda0'       : lambda0,
    'coeffs_z_norm' : coeffs_z_norm,
    'coeffs_f_norm' : coeffs_f_norm,
    'coeffs_rbf_norm': coeffs_rbf_norm,
    'cz_opt'        : cz_opt,
    'cf_opt'        : cf_opt,
    'cr_opt'        : cr_opt,
    'I_opt'         : I_fin,
    'I_des_raw'     : I_des_raw,
    'I_tar'         : I_tar,
    'final_MSE'     : res.fun if 'res' in globals() else best['val']
}

# 2) Write to disk
name='optimization_results7cm633lambda1.pkl'
with open(name, 'wb') as f:
    pickle.dump(vars_to_save, f)

print("All key variables saved to name")