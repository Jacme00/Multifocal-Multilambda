# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 03:46:14 2025

@author: zarkm
"""

from Base import *
import numpy as np
from scipy.special import j0
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

# Normalized random coefficients in [0,1]
coeffs_z_norm = np.random.rand(6)    # 6 Zernike terms
coeffs_f_norm = np.random.rand(100)   # 50 Fourier terms
coeffs_rbf_norm = np.random.rand(10) # 10 RBF terms

# -------------------------------------------------------------------------
# 1) Pack/unpack utilities for your 66-D coefficient vector
# -------------------------------------------------------------------------
def pack(cz, cf, cr):
    return np.concatenate([cz, cf, cr])

def unpack(c_all, Nz, Nf, Nrbf):
    return (c_all[:Nz],
            c_all[Nz:Nz+Nf],
            c_all[Nz+Nf:Nz+Nf+Nrbf])

# -------------------------------------------------------------------------
# 2) Simulation: from coeffs → 1D smoothed through-focus curve I_sm(z)
# -------------------------------------------------------------------------


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

# -------------------------------------------------------------------------
# 3) Objective + finite-difference gradient
# -------------------------------------------------------------------------
def objective(c_all, r, z, R, lambda0, I_des_raw, dims):
    Nz, Nf, Nrbf = dims
    cz, cf, cr = unpack(c_all, Nz, Nf, Nrbf)

    I_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda0)

    # area-match the raw target
    area_sim = np.trapz(I_sim, z)
    area_des = np.trapz(I_des_raw, z)
    I_tar    = I_des_raw * (area_sim/area_des)

    # MSE loss
    return np.mean((I_sim - I_tar)**2)

def fd_grad(c_all, eps, *args):
    f0 = objective(c_all, *args)
    grad = np.zeros_like(c_all)
    for i in range(len(c_all)):
        c_all[i] += eps
        f1 = objective(c_all, *args)
        c_all[i] -= eps
        grad[i] = (f1 - f0) / eps
    return grad

# -------------------------------------------------------------------------
# 4) Main finite-difference GD loop
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # grids and constants
    R       = 4e-3
    Nr      = 200
    r       = np.linspace(0, R, Nr)
    z       = np.linspace(0.03, 0.21, 150)
    lambda0 = 633e-9

    # desired Gaussian focus (sd=1 mm at 7 cm)
    z0      = 0.07
    sigma   = 1e-3
    I_des_raw = np.exp(-0.5 * ((z - z0)/sigma)**2)

    # coefficient dimensions
    dims = (len(coeffs_z_norm),
            len(coeffs_f_norm),
            len(coeffs_rbf_norm))

    # initialize from your current normalized coeffs
    c = pack(coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm)

    # GD settings
    lr    = 1e-3
    eps   = 1e-4
    iters = 200

    history = []
    args = (r, z, R, lambda0, I_des_raw, dims)

    for it in range(iters):
        g = fd_grad(c, eps, *args)
        c -= lr * g
        if it % 10 == 0 or it == iters-1:
            loss = objective(c, *args)
            print(f"Iter {it:3d}  MSE = {loss:.3e}")
            history.append((it, loss))

    # unpack and simulate final
    cz_opt, cf_opt, cr_opt = unpack(c, *dims)
    I_opt = simulate_onaxis(cz_opt, cf_opt, cr_opt, r, z, R, lambda0)

    # area-match and build target for plotting
    area_sim = np.trapz(I_opt, z)
    area_des = np.trapz(I_des_raw, z)
    I_tar    = I_des_raw * (area_sim/area_des)

    # convergence plot
    its, vals = zip(*history)
    plt.figure()
    plt.plot(its, vals, 'o-')
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("Convergence of FD-GD")
    plt.grid(True)
    plt.tight_layout()

    # final focus comparison
    plt.figure(figsize=(8,4))
    plt.plot(z, I_opt, label='Optimized')
    plt.plot(z, I_tar, '--',  label='Desired')
    plt.xlabel('z (m)')
    plt.ylabel('Intensity (arb.)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()


# 4) plot
plt.figure(figsize=(8,4))
plt.plot(z, I_int,   label='Raw integrated')
#plt.plot(z, I_smooth, label='savgol Smoothed', linewidth=2)
plt.plot(z, I_smooth2, label='Gaussianly Smoothed', linewidth=2)
plt.plot(z, I_desired, label='Desired Focus (area-matched)', linestyle='--', linewidth=2)

plt.xlabel('Propagation distance $z$ (m)')
plt.ylabel('Integrated intensity $\int_0^{0.1\,\mathrm{mm}} I(r)\,dr$ (arb. units)')
plt.title('Through‐Focus Intensity Profile')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# --- problem setup (reuse your existing variables/functions) ---
R        = 4e-3
Nr       = 200
r        = np.linspace(0, R, Nr)
z        = np.linspace(0.03, 0.21, 150)
lambda0  = 633e-9

z0       = 0.07
sigma    = 1e-3
I_des_raw= np.exp(-0.5 * ((z - z0)/sigma)**2)

dims     = (len(coeffs_z_norm), len(coeffs_f_norm), len(coeffs_rbf_norm))
total_dim= sum(dims)

def unpack(x):
    Nz, Nf, Nrbf = dims
    return x[:Nz], x[Nz:Nz+Nf], x[Nz+Nf:]

def bb_objective(x):
    cz, cf, cr = unpack(x)
    I_sim = simulate_curve(cz, cf, cr, r, z, R, lambda0)
    area_sim = np.trapz(I_sim, z)
    area_des = np.trapz(I_des_raw, z)
    I_tar    = I_des_raw * (area_sim/area_des)
    return np.mean((I_sim - I_tar)**2)

bounds = [(0,1)] * total_dim

# --- set up interactive plot ---
plt.ion()
fig, ax = plt.subplots(figsize=(8,4))
line_best, = ax.plot([], [], 'C0-', lw=2, label='GA best')
line_des,  = ax.plot(z, I_des_raw, 'k--', lw=1, label='Desired')
ax.set_xlabel('z (m)')
ax.set_ylabel('Intensity (arb.)')
ax.set_title('GA Through-Focus Optimization')
ax.legend()
ax.grid(True)
plt.show()

# --- callback to update plot each generation ---
def ga_callback(xk, convergence):
    cz, cf, cr = unpack(xk)
    I_sim = simulate_curve(cz, cf, cr, r, z, R, lambda0)
    # area-match for fair comparison
    area_sim = np.trapz(I_sim, z)
    area_des = np.trapz(I_des_raw, z)
    I_tar    = I_des_raw * (area_sim/area_des)
    # update line
    line_best.set_data(z, I_sim)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return False   # return False to tell DE to continue

# --- run DE with callback ---
result = differential_evolution(
    bb_objective,
    bounds,
    maxiter=100,
    popsize=15,
    mutation=(0.5,1.0),
    recombination=0.7,
    tol=1e-6,
    disp=True,
    callback=ga_callback,
    polish=True
)

plt.ioff()
print("Done, best MSE:", result.fun)
