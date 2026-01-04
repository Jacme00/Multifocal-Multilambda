# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 04:03:44 2025

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

# --- reuse your problem setup ---
R         = 4e-3
Nr        = 200
r         = np.linspace(0, R, Nr)
z         = np.linspace(0.03, 0.21, 150)
lambda0   = 633e-9
z0, sigma = 0.07, 1e-3
I_des_raw = np.exp(-0.5 * ((z - z0)/sigma)**2)


# Normalized random coefficients in [0,1]
coeffs_z_norm = np.random.rand(6)    # 6 Zernike terms
coeffs_f_norm = np.random.rand(50)   # 50 Fourier terms
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

bounds = [(-100, 100)] * total_dim

# --- set up a generation counter ---
gen_counter = {"it": 0}

def ga_callback(xk, convergence):
    gen_counter["it"] += 1
    best_mse = bb_objective(xk)
    print(f"Generation {gen_counter['it']:3d}: best MSE = {best_mse:.6e}")
    return False  # continue





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
    I_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda0)
    area_sim = np.trapz(I_sim, z)
    area_des = np.trapz(I_des_raw, z)
    I_tar    = I_des_raw * (area_sim/area_des)
    return np.mean((I_sim - I_tar)**2)

bounds = [(-5,5)] * total_dim

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
    I_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda0)
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
    maxiter=10000,
    popsize=50,
    mutation=(0.5,1.0),
    recombination=0.7,
    tol=1e-6,
    disp=True,
    polish=True
)

plt.ioff()
print("Done, best MSE:", result.fun)



























# --- PSO hyperparameters ---
n_particles = 500     # swarm size
n_iters     = 10000     # generations
w_inertia   = 0.1     # inertia weight
c1, c2      = 1.5, 1.5  # cognitive and social coeffs

# --- initialize swarm ---
rng = np.random.default_rng(42)
# positions in [0,1]
pos = rng.random((n_particles, total_dim))*5
# velocities small
vel = np.zeros_like(pos)

# personal bests
pbest_pos = pos.copy()
pbest_val = np.array([bb_objective(p) for p in pos])

# global best
gbest_idx = np.argmin(pbest_val)
gbest_pos = pbest_pos[gbest_idx].copy()
gbest_val = pbest_val[gbest_idx]

print(f"Init: best MSE = {gbest_val:.6e}")

# --- PSO loop ---
for it in range(1, n_iters+1):
    # update velocities and positions
    r1 = rng.random((n_particles, 1))
    r2 = rng.random((n_particles, 1))
    vel = (w_inertia * vel
           + c1 * r1 * (pbest_pos - pos)
           + c2 * r2 * (gbest_pos - pos))
    pos += vel
    # clamp to [0,1]
    np.clip(pos, 0.0, 1.0, out=pos)

    # evaluate
    vals = np.array([bb_objective(p) for p in pos])
    # update personal bests
    better = vals < pbest_val
    pbest_pos[better] = pos[better]
    pbest_val[better] = vals[better]

    # update global best
    min_idx = np.argmin(pbest_val)
    if pbest_val[min_idx] < gbest_val:
        gbest_val = pbest_val[min_idx]
        gbest_pos = pbest_pos[min_idx].copy()

    # print progress
    if it % 10 == 0 or it == 1 or it == n_iters:
        print(f"Iter {it:3d}: best MSE = {gbest_val:.6e}")

# --- unpack and report final result ---
cz_opt, cf_opt, cr_opt = unpack(gbest_pos)
I_opt = simulate_onaxis(cz_opt, cf_opt, cr_opt, r, z, R, lambda0)

# area-match target for plotting or further use
area_opt = np.trapz(I_opt, z)
I_tar   = I_des_raw * (area_opt / np.trapz(I_des_raw, z))

print("\nPSO finished. Final best MSE =", gbest_val)