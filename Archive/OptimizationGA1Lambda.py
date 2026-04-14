# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 17:39:36 2025

@author: zarkm
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 04:21:28 2025

@author: zarkm
"""

from math import factorial
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
coeffs_f_norm = np.random.rand(20)   # 50 Fourier terms
coeffs_rbf_norm = np.random.rand(10) # 10 RBF terms


dims      = (len(coeffs_z_norm), len(coeffs_f_norm), len(coeffs_rbf_norm))
total_dim = sum(dims)



def zernike_expand_true(coeffs_z_norm, r, R):
    """
    True radial Zernike expansion with normalized coefficients [0,1].
    coeffs_z_norm: array of size K, values in [0,1]
    Uses internal default range_z to map to actual amplitudes.
    """
    range_z = 0.1  # max absolute amplitude for Zernike terms
    # Map normalized to actual range [-range_z, +range_z]
    coeffs_z = (2 * coeffs_z_norm - 1) * range_z
    rho = r / R
    z = np.zeros_like(r)
    for i, c in enumerate(coeffs_z):
        n = 2 * i  # radial order: 0,2,4,...
        Rn = np.zeros_like(rho)
        for k in range(n // 2 + 1):
            num = (-1)**k * factorial(n - k)
            den = factorial(k) * factorial((n // 2 - k))**2
            Rn += (num / den) * rho**(n - 2 * k)
        z += c * Rn
    return z

def fourier_expand(coeffs_f_norm, r, R):
    """
    Fourier/cosine series with normalized coefficients [0,1].
    coeffs_f_norm: array of size M, values in [0,1]
    Uses internal default range_f to map to actual amplitudes.
    """
    range_f = 0.05  # max absolute amplitude for Fourier terms
    coeffs_f = (2 * coeffs_f_norm - 1) * range_f
    z = np.zeros_like(r)
    for m, a in enumerate(coeffs_f, start=1):
        z += a * np.cos(m * np.pi * r / R)
    return z

def rbf_expand(coeffs_rbf_norm, r, R):
    """
    Gaussian RBFs with normalized weights [0,1].
    coeffs_rbf_norm: array of size J, values in [0,1]
    Uses internal defaults range_rbf and sigma_rbf.
    """
    range_rbf = 0.2   # max absolute weight for RBFs
    sigma_rbf = 0.05  # width of each Gaussian
    coeffs_rbf = (2 * coeffs_rbf_norm - 1) * range_rbf
    centers = np.linspace(0, R * (1 - 1 / len(coeffs_rbf)), len(coeffs_rbf))
    z = np.zeros_like(r)
    for w, rc in zip(coeffs_rbf, centers):
        z += w * np.exp(-0.5 * ((r - rc) / sigma_rbf)**2)
    return z

def abstract_profile_normalized(r, coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm, R):
    """
    Generate a deterministic, smooth radial profile using normalized coeffs [0,1].
    Uses internal defaults for range_z, range_f, range_rbf, sigma_rbf.
    """
    h = np.zeros_like(r)
    h += zernike_expand_true(coeffs_z_norm, r, R)
    h += fourier_expand(coeffs_f_norm, r, R)
    h += rbf_expand(coeffs_rbf_norm, r, R)
    # Enforce non-negativity
    h -= h.min()
    return h

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


# --- 2) Run differential evolution ---
bounds = [(0,1)] * total_dim
result = differential_evolution(
    bb_objective,
    bounds,
    strategy='best1bin',
    maxiter=2000,        # number of generations
    popsize=5,         # population size = 15*dim
    tol=1e-3,
    mutation=(0.05,0.2),
    recombination=0.3,
    disp=True,
    polish=True         # final local refine
)



print("\nGenetic optimization finished.")
print(" Success:", result.success)
print(f" Final best MSE = {best['val']:.6e}")
x_opt = result.x

# --- 3) Extract and plot final curves ---
cz_opt, cf_opt, cr_opt = unpack(x_opt)
I_opt = simulate_onaxis(cz_opt, cf_opt, cr_opt, r, z, R, lambda0)
area_sim = np.trapz(I_opt, z)
I_fin    = I_des_raw * (area_sim / np.trapz(I_des_raw, z))

plt.figure(figsize=(8,4))
plt.plot(z, I_opt, 'C0-', lw=2, label='GA Best')
plt.plot(z, I_fin, 'k--', lw=1, label='Target (area‐matched)')
plt.xlabel('z (m)')
plt.ylabel('Intensity (arb.)')
plt.title('Final Through-Focus Profile')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()