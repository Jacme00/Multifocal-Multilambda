# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 11:44:14 2025

@author: zarkm
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 11:44:14 2025
@author: zarkm
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from math import factorial

# -----------------------------------------------------------------------------
# 1) Torch re-implementations of your basis expansions
# -----------------------------------------------------------------------------
def zernike_expand_true_torch(cz, r, R):
    device = r.device
    range_z = 0.1
    coeffs_z = (2*cz - 1)*range_z
    rho = r / R
    out = torch.zeros_like(rho)
    K = cz.shape[0]
    for i in range(K):
        c = coeffs_z[i]
        n = 2*i
        Rn = torch.zeros_like(rho)
        for k in range(n//2+1):
            num = (-1)**k * factorial(n-k)
            den = factorial(k)*factorial(n//2-k)**2
            Rn = Rn + (num/den)*rho.pow(n-2*k)
        out = out + c*Rn
    return out

def fourier_expand_torch(cf, r, R):
    device = r.device
    range_f = 0.05
    coeffs_f = (2*cf - 1)*range_f
    out = torch.zeros_like(r)
    for m in range(1, coeffs_f.shape[0]+1):
        out = out + coeffs_f[m-1]*torch.cos(m*torch.pi*r/R)
    return out

def rbf_expand_torch(cr, r, R):
    device = r.device
    range_rbf = 0.2
    sigma_rbf = 0.05*R
    coeffs_rbf = (2*cr - 1)*range_rbf
    J = coeffs_rbf.shape[0]
    centers = torch.linspace(0, R*(1-1/J), J, device=device)
    out = torch.zeros_like(r)
    for j in range(J):
        out = out + coeffs_rbf[j]*torch.exp(-0.5*((r-centers[j])/sigma_rbf)**2)
    return out

def abstract_profile_torch(r, cz, cf, cr, R):
    h = zernike_expand_true_torch(cz, r, R)
    h = h + fourier_expand_torch(cf, r, R)
    h = h + rbf_expand_torch(cr, r, R)
    return h - h.min()

# -----------------------------------------------------------------------------
# 2) Problem setup (CPU)
# -----------------------------------------------------------------------------
R         = 4e-3
Nr        = 200
r         = np.linspace(0, R, Nr)
z         = np.linspace(0.03, 0.21, 150)
lambda0   = 633e-9
z0, sigma = 0.07, 1e-3
I_des_raw = np.exp(-0.5*((z-z0)/sigma)**2)

coeffs_z_norm   = np.random.rand(6)
coeffs_f_norm   = np.random.rand(60)
coeffs_rbf_norm = np.random.rand(10)
dims      = (len(coeffs_z_norm), len(coeffs_f_norm), len(coeffs_rbf_norm))
total_dim = sum(dims)

# -----------------------------------------------------------------------------
# 3) GPU setup
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

r_t   = torch.linspace(0, R, steps=Nr, device=device)
z_t   = torch.linspace(0.03, 0.21, steps=z.size, device=device)
lam_t = torch.tensor(lambda0, device=device)
k_t   = 2*torch.pi/lam_t

# -----------------------------------------------------------------------------
# 4) On-axis simulator (returns GPU tensor)
# -----------------------------------------------------------------------------

def simulate_onaxis_cuda(cz_t, cf_t, cr_t):
    """
    Fully‐vectorized on‐axis simulator on the GPU.
    Inputs:
      cz_t, cf_t, cr_t: 1D torch.Tensor on `device` of lengths Nzern, Nfourier, Nrbf
    Globals:
      r_t   : (Nr,) torch.linspace on device
      z_t   : (Nz,) torch.linspace on device
      R     : scalar radius (float)
      lam_t : scalar wavelength (torch scalar on device)
      k_t   : scalar wavenumber (torch scalar on device)
    Returns:
      I_on : (Nz,) torch.Tensor of on-axis intensities, on device
    """
    # 1) build phase φ(r) = 2π·h(r) entirely in Torch
    h_t   = abstract_profile_torch(r_t, cz_t, cf_t, cr_t, R)  # (Nr,)
    phi_t = 2*torch.pi * h_t                                  # (Nr,)

    # 2) expand dims for broadcasting:
    #    R1: (1,1,Nr)   → input radius
    #    R2: (1,Nr,1)   → output radius (unused for on-axis but needed for kernel)
    #    Z:  (Nz,1,1)   → propagation distances
    R1 = r_t.view(1,1,-1)
    R2 = r_t.view(1,-1,1)
    Z  = z_t.view(-1,1,1)
    PH = phi_t.view(1,1,-1)

    # 3) prefactor and quadratic phase Q
    prefac = torch.exp(1j * k_t * Z) / (1j * lam_t * Z)        # (Nz,1,1)
    Q      = torch.exp(1j * k_t * (R1**2) / (2 * Z))           # (Nz,1,Nr)

    # 4) Hankel kernel via Bessel J0
    K = torch.special.bessel_j0(k_t * R2 * R1 / Z)            # (Nz,Nr,Nr)

    # 5) integrand = exp(iφ) * Q * K * r_in
    integrand = torch.exp(1j * PH) * Q * K * R1               # (Nz,Nr,Nr)

    # 6) first radial integral: E(z,r_out) = prefac * ∫ integrand dr_in
    E = prefac * torch.trapz(integrand, r_t, dim=2)           # (Nz,Nr)

    # 7) intensity I(z,r_out) = |E|^2
    I = torch.abs(E)**2                                       # (Nz,Nr)

    # 8) radial cutoff integration to get on-axis (r≤0.1 mm)
    mask = (r_t <= 0.1e-3).to(I.dtype).view(1,-1)              # (1,Nr)
    dr   = r_t[1] - r_t[0]                                    # scalar spacing
    I_on = torch.trapz(I * mask, dx=dr, dim=1)                # (Nz,)

    return I_on  # still on device
# -----------------------------------------------------------------------------
# 5) Black-box objective
# -----------------------------------------------------------------------------
def unpack(x):
    Nz, Nf, Nrbf = dims
    return x[:Nz], x[Nz:Nz+Nf], x[Nz+Nf:]

def bb_objective(x):
    cz, cf, cr = unpack(x)
    cz_t = torch.tensor(cz, dtype=torch.float32, device=device)
    cf_t = torch.tensor(cf, dtype=torch.float32, device=device)
    cr_t = torch.tensor(cr, dtype=torch.float32, device=device)

    # 1) forward on GPU
    I_on_t = simulate_onaxis_cuda(cz_t, cf_t, cr_t)

    # 2) back to CPU+NumPy for smoothing & loss
    I_on = I_on_t.detach().cpu().numpy()
    I_sm = gaussian_filter1d(I_on, sigma=1.0)

    area_sim = np.trapz(I_sm, z)
    area_des = np.trapz(I_des_raw, z)
    I_tar    = I_des_raw*(area_sim/area_des)

    return np.mean((I_sm - I_tar)**2)

# -----------------------------------------------------------------------------
# 6) Nelder-Mead pattern search
# -----------------------------------------------------------------------------
x0 = np.concatenate([coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm])
best = {"val": bb_objective(x0)}
cnt  = {"it": 0}

def nm_callback(xk):
    cnt["it"] += 1
    val = bb_objective(xk)
    if val < best["val"]:
        best["val"] = val
        print(f"Iter {cnt['it']:4d}: new best MSE = {val:.6e}")

print("Starting MSE =", best["val"])
res = minimize(
    bb_objective, x0,
    method='Nelder-Mead',
    callback=nm_callback,
    options={'maxiter':500,'xatol':1e-3,'fatol':1e-3,'disp':True}
)

print("\nDone. Success:", res.success, "Final MSE:", res.fun)

# -----------------------------------------------------------------------------
# 7) Final simulate + plot + save
# -----------------------------------------------------------------------------
cz_opt, cf_opt, cr_opt = unpack(res.x)
I_opt_t = simulate_onaxis_cuda(
    torch.tensor(cz_opt, device=device),
    torch.tensor(cf_opt, device=device),
    torch.tensor(cr_opt, device=device)
)

# convert for plotting
I_opt = I_opt_t.detach().cpu().numpy()
area_sim = np.trapz(I_opt, z)
I_tar    = I_des_raw*(area_sim/np.trapz(I_des_raw,z))

plt.figure(figsize=(8,4))
plt.plot(z, I_opt, 'C0-', lw=2, label='Optimized')
plt.plot(z, I_tar, 'k--', lw=1, label='Desired')
plt.xlabel('z (m)'); plt.ylabel('I (arb.)')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

vars_to_save = {
    'r': r, 'z': z, 'R': R, 'lambda0': lambda0,
    'cz_opt': cz_opt, 'cf_opt': cf_opt, 'cr_opt': cr_opt,
    'I_opt': I_opt, 'I_des_raw': I_des_raw, 'I_tar': I_tar,
    'final_MSE': res.fun
}
with open('optim_results_gpu.pkl','wb') as f:
    pickle.dump(vars_to_save, f)
print("Saved to optim_results_gpu.pkl")
