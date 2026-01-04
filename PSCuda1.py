# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:09:38 2025

@author: zarkm
"""

# -*- coding: utf-8 -*-
"""
GPU-accelerated on-axis pattern search for custom DOE focus design
"""

import numpy as np
import torch
from Base import abstract_profile_normalized
from scipy.special import j0
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle

# -----------------------------------------------------------------------------
# 1) Setup: CPU arrays and GPU device
# -----------------------------------------------------------------------------
R         = 4e-3
Nr        = 200
r         = np.linspace(0, R, Nr)
z         = np.linspace(0.03, 0.21, 150)
lambda0   = 633e-9
z0, sigma = 0.07, 1e-3
I_des_raw = np.exp(-0.5 * ((z - z0)/sigma)**2)

# normalized-random initial coeffs
coeffs_z_norm   = np.random.rand(6)
coeffs_f_norm   = np.random.rand(60)
coeffs_rbf_norm = np.random.rand(10)

dims      = (len(coeffs_z_norm), len(coeffs_f_norm), len(coeffs_rbf_norm))
total_dim = sum(dims)

# choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------------------------------------------------------
# 2) Move constants to GPU as torch tensors
# -----------------------------------------------------------------------------
r_t    = torch.from_numpy(r).float().to(device)      # (Nr,)
z_t    = torch.from_numpy(z).float().to(device)      # (Nz,)
R_t    = torch.tensor(R,    device=device)          # scalar
lam_t  = torch.tensor(lambda0, device=device)       # scalar
k_t    = 2*torch.pi / lam_t                         # scalar

# -----------------------------------------------------------------------------
# 3) Pack/unpack utilities
# -----------------------------------------------------------------------------
def unpack(x):
    Nz, Nf, Nrbf = dims
    return x[:Nz], x[Nz:Nz+Nf], x[Nz+Nf:]

# -----------------------------------------------------------------------------
# 4) GPU-accelerated on-axis simulator
# -----------------------------------------------------------------------------
def simulate_onaxis_cuda(cz_t, cf_t, cr_t, sigma_smooth=1.0):
    # 4.1) build radial profile h(r) via Base (NumPy) → torch
    h_np = abstract_profile_normalized(
        r / R, 
        cz_t.cpu().numpy(), 
        cf_t.cpu().numpy(), 
        cr_t.cpu().numpy(), 
        1.0
    )
    h_t   = torch.from_numpy(h_np).float().to(device)  # (Nr,)
    phi_t = 2*torch.pi * h_t                          # (Nr,)

    Nz = z_t.shape[0]
    I_on = torch.empty(Nz, device=device)

    # 4.2) loop over z planes on GPU
    for iz in range(Nz):
        zi     = z_t[iz]
        prefac = torch.exp(1j * k_t * zi) / (1j * lam_t * zi)
        Q      = torch.exp(1j * k_t * r_t**2 / (2*zi))
        integrand = torch.exp(1j*phi_t) * Q * r_t
        E0     = prefac * torch.trapz(integrand, r_t)
        I_on[iz] = torch.abs(E0)**2

    # 4.3) smooth result on CPU
    I_cpu = I_on.detach().cpu().numpy()
    #I_sm  = gaussian_filter1d(I_cpu, sigma=sigma_smooth)
    return I_cpu

# -----------------------------------------------------------------------------
# 5) Black-box objective calling the GPU simulator
# -----------------------------------------------------------------------------
def bb_objective(x):
    cz, cf, cr = unpack(x)
    # to torch
    cz_t = torch.tensor(cz, dtype=torch.float32, device=device)
    cf_t = torch.tensor(cf, dtype=torch.float32, device=device)
    cr_t = torch.tensor(cr, dtype=torch.float32, device=device)

    I_sim = simulate_onaxis_cuda(cz_t, cf_t, cr_t, sigma_smooth=1.0)

    # area-match against raw target
    area_sim = np.trapz(I_sim, z)
    area_des = np.trapz(I_des_raw, z)
    I_tar    = I_des_raw * (area_sim/area_des)

    return np.mean((I_sim - I_tar)**2)

# -----------------------------------------------------------------------------
# 6) Pattern search: Nelder–Mead with progress prints
# -----------------------------------------------------------------------------
x0 = np.concatenate([coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm])
best = {"val": bb_objective(x0)}
counter = {"it": 0}

def nm_callback(xk):
    counter["it"] += 1
    val = bb_objective(xk)
    if val < best["val"]:
        best["val"] = val
        print(f"Iter {counter['it']:4d}: new best MSE = {val:.6e}")
    return

print("Starting MSE:", best["val"])
res = minimize(
    bb_objective,
    x0,
    method='Nelder-Mead',
    callback=nm_callback,
    options={
        'maxiter': 500,
        'xatol':   1e-3,
        'fatol':   1e-3,
        'disp':    True
    }
)

print("\nOptimization finished.")
print(" Success:", res.success)
print(" Final MSE =", res.fun)

# -----------------------------------------------------------------------------
# 7) Extract final result and save variables
# -----------------------------------------------------------------------------
cz_opt, cf_opt, cr_opt = unpack(res.x)
I_opt = simulate_onaxis_cuda(
    torch.tensor(cz_opt, device=device),
    torch.tensor(cf_opt, device=device),
    torch.tensor(cr_opt, device=device),
    sigma_smooth=1.0
)

# area-match for final target
area_sim = np.trapz(I_opt, z)
I_tar    = I_des_raw * (area_sim / np.trapz(I_des_raw, z))

# plot
plt.figure(figsize=(8,4))
plt.plot(z, I_opt, 'C0-', lw=2, label='Optimized')
plt.plot(z, I_tar, 'k--', lw=1, label='Desired')
plt.xlabel('Propagation distance $z$ (m)')
plt.ylabel('On-axis intensity (arb. u.)')
plt.title('Through-Focus Profile (On-Axis)')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

# save key variables
vars_to_save = {
    'r': r, 'z': z, 'R': R, 'lambda0': lambda0,
    'coeffs_z_norm': coeffs_z_norm,
    'coeffs_f_norm': coeffs_f_norm,
    'coeffs_rbf_norm': coeffs_rbf_norm,
    'cz_opt': cz_opt, 'cf_opt': cf_opt, 'cr_opt': cr_opt,
    'I_opt': I_opt, 'I_des_raw': I_des_raw, 'I_tar': I_tar,
    'final_MSE': res.fun
}
with open('optimization_results_gpu.pkl', 'wb') as f:
    pickle.dump(vars_to_save, f)
print("Results saved to optimization_results_gpu.pkl")
