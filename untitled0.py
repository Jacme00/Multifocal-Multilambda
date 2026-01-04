# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:27:15 2025

@author: zarkm
"""

import numpy as np
from math import factorial
from scipy.special import j0
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1) Basis‑expansion routines (normalized coeffs in [0,1] → physical h(r))
# -----------------------------------------------------------------------------
def zernike_expand_true(coeffs_z_norm, r, R):
    range_z = 0.1
    coeffs_z = (2 * coeffs_z_norm - 1) * range_z
    rho = r / R
    z = np.zeros_like(r)
    for i, c in enumerate(coeffs_z):
        n = 2 * i
        Rn = np.zeros_like(rho)
        for k in range(n // 2 + 1):
            num = (-1)**k * factorial(n - k)
            den = factorial(k) * factorial((n // 2 - k))**2
            Rn += (num / den) * rho**(n - 2 * k)
        z += c * Rn
    return z

def fourier_expand(coeffs_f_norm, r, R):
    range_f = 0.05
    coeffs_f = (2 * coeffs_f_norm - 1) * range_f
    z = np.zeros_like(r)
    for m, a in enumerate(coeffs_f, start=1):
        z += a * np.cos(m * np.pi * r / R)
    return z

def rbf_expand(coeffs_rbf_norm, r, R):
    range_rbf = 0.2
    sigma_rbf = 0.05 * R
    coeffs_rbf = (2 * coeffs_rbf_norm - 1) * range_rbf
    centers = np.linspace(0, R * (1 - 1 / len(coeffs_rbf)), len(coeffs_rbf))
    z = np.zeros_like(r)
    for w, rc in zip(coeffs_rbf, centers):
        z += w * np.exp(-0.5 * ((r - rc) / sigma_rbf)**2)
    return z

def abstract_profile_normalized(r, coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm, R):
    """Combine all three bases and enforce non‑negativity."""
    h = np.zeros_like(r)
    h += zernike_expand_true(coeffs_z_norm, r, R)
    h += fourier_expand(coeffs_f_norm, r, R)
    h += rbf_expand(coeffs_rbf_norm, r, R)
    h -= h.min()  # shift to ≥0
    return h

# -----------------------------------------------------------------------------
# 2) On‑axis field via a Hankel integral at r=0
# -----------------------------------------------------------------------------
def hankel_on_axis(phi, r, wavelength, z):
    """
    Compute E(0,z) = ∫0^R exp(i*phi(r)) * exp(i k r^2/(2z)) * r dr,
    times the prefactor exp(i kz)/(i λ z).
    """
    k = 2 * np.pi / wavelength
    prefac = np.exp(1j * k * z) / (1j * wavelength * z)
    Q = np.exp(1j * k * r**2 / (2 * z))
    integrand = np.exp(1j * phi) * Q * r
    return prefac * np.trapz(integrand, r)

# -----------------------------------------------------------------------------
# 3) Pack/unpack utilities for the 66‑D vector
# -----------------------------------------------------------------------------
def pack(c_z, c_f, c_r):
    return np.concatenate([c_z, c_f, c_r])

def unpack(c_all, Nz, Nf, Nrbf):
    c_z = c_all[:Nz]
    c_f = c_all[Nz:Nz+Nf]
    c_r = c_all[Nz+Nf:Nz+Nf+Nrbf]
    return c_z, c_f, c_r

# -----------------------------------------------------------------------------
# 4) Objective and finite‑difference gradient
# -----------------------------------------------------------------------------
def objective(c_all, r, R, wavelength, z0, dims):
    Nz, Nf, Nrbf = dims
    c_z, c_f, c_r = unpack(c_all, Nz, Nf, Nrbf)
    h = abstract_profile_normalized(r, c_z, c_f, c_r, R)
    phi = 2 * np.pi * h
    E0 = hankel_on_axis(phi, r, wavelength, z0)
    I0 = np.abs(E0)**2
    return -I0  # we minimize negative intensity

def fd_grad(c_all, eps, *obj_args):
    f0 = objective(c_all, *obj_args)
    grad = np.zeros_like(c_all)
    for i in range(len(c_all)):
        c_all[i] += eps
        f1 = objective(c_all, *obj_args)
        c_all[i] -= eps
        grad[i] = (f1 - f0) / eps
    return grad

# -----------------------------------------------------------------------------
# 5) Main optimization loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Parameters ---
    R = 4e-3               # DOE radius (4 mm)
    Nr = 200               # radial samples
    r = np.linspace(0, R, Nr)
    λ0 = 633e-9            # wavelength (633 nm)
    z0 = 0.12              # target focus (12 cm)
    dims = (6, 50, 10)     # #Zernike, #Fourier, #RBF = 66 total

    # --- Initialization ---
    np.random.seed(42)
    coeffs_z = np.random.rand(dims[0])
    coeffs_f = np.random.rand(dims[1])
    coeffs_rbf = np.random.rand(dims[2])
    c = pack(coeffs_z, coeffs_f, coeffs_rbf)

    # --- GD settings ---
    lr = 2e-5
    eps = 1e-5
    iters = 20000

    # --- Run optimization ---
    history = []
    obj_args = (r, R, λ0, z0, dims)

    for it in range(iters):
        g = fd_grad(c, eps, *obj_args)
        c -= lr * g
        if it % 20 == 0 or it == iters-1:
            I0 = -objective(c, *obj_args)
            print(f"Iter {it:3d}: I(0,{z0:.2f} m) = {I0:.4f}")
            history.append((it, I0))

    # --- Plot convergence ---
    its, vals = zip(*history)
    plt.figure()
    plt.plot(its, vals, 'o-')
    plt.xlabel("Iteration")
    plt.ylabel("On‑axis Intensity @ 12 cm")
    plt.title("FD‑GD Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Optional: visualize final phase map ---
    coeffs_z_opt, coeffs_f_opt, coeffs_rbf_opt = unpack(c, *dims)
    h_opt = abstract_profile_normalized(r, coeffs_z_opt, coeffs_f_opt, coeffs_rbf_opt, R)
    phi_opt = 2 * np.pi * h_opt
    plt.figure()
    plt.plot(r*1e3, phi_opt)
    plt.xlabel("r (mm)")
    plt.ylabel("Phase φ(r) [rad]")
    plt.title("Optimized Radial Phase Profile")
    plt.tight_layout()
    plt.show()
