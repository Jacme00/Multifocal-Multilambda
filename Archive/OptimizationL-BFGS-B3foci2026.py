# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 05:09:53 2026

@author: zarkm
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 20:32:23 2026

@author: zarkm
"""

import time
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

# ------
R         = 4e-3
Nr        = 600
r         = np.linspace(0, R, Nr)
z         = np.linspace(0.03, 0.21, 150)
lambda0   = 633e-9
lambda1   = 532e-9
lambda2   = 445e-9

z0, z1, z2, sigma, sigma1, sigma2 = 0.07, 0.11, 0.17, 1e-3, 1e-3, 1e-3
I_des_raw = np.exp(-0.5 * ((z - z0)/sigma)**2)
I_des1_raw = np.exp(-0.5 * ((z - z1)/sigma1)**2) 
I_des2_raw = np.exp(-0.5 * ((z - z2)/sigma2)**2) #added yet another new lambdas, new sigmas and new desired target intensity profiles
area0_des = np.trapz(I_des_raw, z)
area1_des = np.trapz(I_des1_raw, z)
area2_des = np.trapz(I_des2_raw, z)


# Normalized random coefficients in [0,1]
coeffs_z_norm = np.random.rand(6)    # 6 Zernike terms
coeffs_f_norm = np.random.rand(300)   #  Fourier terms
coeffs_rbf_norm = np.random.rand(7) #  RBF terms


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


def revolve_profile_to_2D(h, R, N):
    """
    Revolve a 1D radial profile h(r) into a 2D circularly symmetric DOE of size N x N.
    Points with radius > R are set to zero.
    """
    # Create 2D grid
    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    X, Y = np.meshgrid(x, y)
    # Compute radial coordinate
    rho = np.sqrt(X**2 + Y**2)
    # Interpolate h(r) for each rho
    h2D = np.interp(rho, np.linspace(0, R, len(h)), h, left=0, right=0)
    return h2D

def unpack(x):
    Nz, Nf, Nrbf = dims
    return x[:Nz], x[Nz:Nz+Nf], x[Nz+Nf:]


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


# def bb_objective(x):
#     """
#     changed the objective function adding the intensity profile contribution from a new lambda
#     """
#     cz, cf, cr = unpack(x)

#     I0_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda0)
#     I1_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda1)

#     area0_sim = np.trapz(I0_sim, z)
#     area1_sim = np.trapz(I1_sim, z)

#     I0_tar = I_des_raw  * (area0_sim / area0_des)
#     I1_tar = I_des1_raw * (area1_sim / area1_des)

#     L0 = np.mean((I0_sim - I0_tar)**2)
#     L1 = np.mean((I1_sim - I1_tar)**2)

#     return L0 + L1


def bb_objective(x):
    
    """
    changed the objective function adding the intensity profile contribution from a new lambda
    """
    cz, cf, cr = unpack(x)
    I0_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda0)
    I1_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda1)
    I2_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda2)
    
    area0_sim = np.trapz(I0_sim, z)
    area1_sim = np.trapz(I1_sim, z)
    area2_sim = np.trapz(I2_sim, z)
    
    eps = 1e-12

    # Normalize simulated curves by their own areas
    I0n = I0_sim / (area0_sim + eps)
    I1n = I1_sim / (area1_sim + eps)
    I2n = I2_sim / (area2_sim + eps)

    # Normalize desired curves by their (constant) areas
    T0n = I_des_raw  / (area0_des + eps)
    T1n = I_des1_raw / (area1_des + eps)
    T2n = I_des2_raw / (area2_des + eps)
    
    
    L0 = np.mean((I0n - T0n)**2)
    L1 = np.mean((I1n - T1n)**2)
    L2 = np.mean((I2n - T2n)**2)
    return L0 + L1 + L2

# initial guess
x0 = np.concatenate([coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm])

# callback to print progress
best = {"val": bb_objective(x0)}
counter = {"it": 0}
last_time = time.perf_counter()
def powell_callback(xk):
    global last_time
    counter["it"] += 1

    # only act every 1 000 iterations
    if counter["it"] % 1000 != 0:
        return

    # compute elapsed
    now    = time.perf_counter()
    dt     = now - last_time
    last_time = now

    # evaluate objective and possibly print new best
    val = bb_objective(xk)
    if val < best["val"]:
        best["val"] = val
        print(f"Iter {counter['it']:6d}: New best MSE = {val:.6e}  (Δt = {dt:.1f}s)")
    else:
        print(f"Iter {counter['it']:6d}: best MSE still {best['val']:.6e}  (Δt = {dt:.1f}s)")



t0 = time.perf_counter()
bb_objective(x0)
print("bb_objective time =", time.perf_counter() - t0)


# --- 1) Start both curves at zero ---
I0_best = np.zeros_like(z)
I0_tar  = np.zeros_like(z)

# --- 3) Tracking state ---
best = {
    'val': bb_objective(x0),   # best‐so‐far loss
    'y'  : I0_best.copy()      # best‐so‐far sim curve
}
counter = {'it': 0}

# # 4) Run Nelder–Mead with this callback
# res = minimize(
#     bb_objective,
#     x0,
#     method='Nelder-Mead',
#     callback=powell_callback,
#     options={
#         'maxiter': 2000000,
#         'xatol':   1e-10,
#         'fatol':   1e-10,
#         'disp':    False
#     }
# )


# bounds = [(0.0, 1.0)] * total_dim

# res_de = differential_evolution(
#     bb_objective,
#     bounds=bounds,
#     maxiter=100,      
#     popsize=30,       
#     polish=False,
#     updating="deferred",
#     workers=1, 
#     disp=True         
# )


# res = minimize(
#     bb_objective,
#     res_de.x,
#     method="L-BFGS-B",
#     bounds=bounds,
#     options={"maxiter": 2000, "disp": True}
# )



bounds = [(0.0, 1.0)] * total_dim

res = minimize(
    bb_objective,
    x0,
    method="L-BFGS-B",
    bounds=bounds,
    options={
        "maxiter": 200,
        "maxfun": 200000,
        "disp": True
    }
)


print(res.message)
print("nit:", getattr(res, "nit", None))
print("nfev:", getattr(res, "nfev", None))

plt.ioff()

print("\nPattern search (Powell) finished.")
print("  Success:", res.success)
print(f"  Final best MSE = {res.fun:.6e}")

# unpack final
cz_opt, cf_opt, cr_opt = unpack(res.x)
I_opt1 = simulate_onaxis(cz_opt, cf_opt, cr_opt, r, z, R, lambda0)
I_opt2 = simulate_onaxis(cz_opt, cf_opt, cr_opt, r, z, R, lambda1)
I_opt3 = simulate_onaxis(cz_opt, cf_opt, cr_opt, r, z, R, lambda2)
# (You can now plot I_opt vs. the target as before.)


area_sim1 = np.trapz(I_opt1, z)
area_sim2 = np.trapz(I_opt2, z)
area_sim3 = np.trapz(I_opt3, z)

area_des1 = np.trapz(I_des_raw, z)
area_des2 = np.trapz(I_des1_raw, z)
area_des3 = np.trapz(I_des2_raw, z)

I_fin1    = I_des_raw * (area_sim1/area_des1)
I_fin2    = I_des1_raw * (area_sim2/area_des2)
I_fin3    = I_des2_raw * (area_sim3/area_des3)


# 4) plot
plt.figure(figsize=(8,4))
plt.plot(z, I_opt1,   label='Best1')
plt.plot(z, I_opt2,   label='Best2')
plt.plot(z, I_opt3,   label='Best3')

#plt.plot(z, I_smooth, label='savgol Smoothed', linewidth=2)
#plt.plot(z, I_opt, label='Gaussianly Smoothed', linewidth=2)
plt.plot(z, I_fin1, label='Desired Focus1 (area-matched)', linestyle='--', linewidth=2)
plt.plot(z, I_fin2, label='Desired Focus2 (area-matched)', linestyle='--', linewidth=2)
plt.plot(z, I_fin3, label='Desired Focus3 (area-matched)', linestyle='--', linewidth=2)

plt.xlabel('Propagation distance $z$ (m)')
plt.ylabel('Intensity I(r)\,dr$ (arb. units)')
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

# Revolve to 2D DOE
h2D = revolve_profile_to_2D(h, R, Nr)

# Display the 2D DOE phase map
plt.figure(figsize=(5, 5))
plt.imshow(h2D, extent=[-R, R, -R, R], origin='lower')
plt.colorbar(label='Height (arb. units)')
plt.title('Circularly Symmetric DOE from Radial Profile')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()





# 1) Gather everything you actually produced into a dict
vars_to_save = {
    # grids & physical parameters
    'r'        : r,
    'z'        : z,
    'R'        : R,
    'lambda0'  : lambda0,
    'lambda1'  : lambda1,
    'lambda2'  : lambda2,

    # desired target profiles
    'z0'        : z0,
    'z1'        : z1,
    'z2'        : z2,
    'sigma'     : sigma,
    'sigma1'    : sigma1,
    'sigma2'    : sigma2,
    'I_des0'    : I_des_raw,
    'I_des1'    : I_des1_raw,
    'I_des2'    : I_des2_raw,
    
    
    
    
    # optimized coefficients
    'cz_opt'   : cz_opt,
    'cf_opt'   : cf_opt,
    'cr_opt'   : cr_opt,

    # optimized simulated curves
    'I_opt_lambda0' : I_opt1,
    'I_opt_lambda1' : I_opt2,
    'I_opt_lambda2' : I_opt3,
    
    
    # area-matched target curves used for plotting
    'I_tar_lambda0' : I_fin1,
    'I_tar_lambda1' : I_fin2,
    'I_tar_lambda2' : I_fin3,

    # final objective value
    'final_loss' : res.fun
}


# 2) Write to disk
name='optimization_results3lambdas.pkl'
with open(name, 'wb') as f:
    pickle.dump(vars_to_save, f)

print("All key variables saved to name")
