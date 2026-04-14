# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 20:35:29 2025

@author: zarkm
"""
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

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


def generate_profile(r, coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm, R):
    """
    Abstract radial profile as before (deterministic, normalized coeffs).
    Uses hard-coded ranges inside each expand function.
    """
    # Reuse the previously defined functions (assuming they are in scope)
    h = abstract_profile_normalized(r, coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm, R)
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

# Example usage
R = 1.0
N = 500
r = np.linspace(0, R, N)

# Normalized random coefficients in [0,1]
coeffs_z_norm = np.random.rand(6)    # 6 Zernike terms
coeffs_f_norm = np.random.rand(50)   # 50 Fourier terms
coeffs_rbf_norm = np.random.rand(10) # 10 RBF terms

h = abstract_profile_normalized(r, coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm, R)

plt.figure(figsize=(6, 4))
plt.plot(r, h, color='darkorange')
plt.xlabel("Normalized Radius r")
plt.ylabel("Height h(r)")
plt.title("Profile from Coeffs")
plt.grid(True)
plt.tight_layout()
plt.show()



# Generate radial profile
h = generate_profile(r, coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm, R)

# Revolve to 2D DOE
h2D = revolve_profile_to_2D(h, R, N)

# Display the 2D DOE phase map
plt.figure(figsize=(5, 5))
plt.imshow(h2D, extent=[-R, R, -R, R], origin='lower')
plt.colorbar(label='Height (arb. units)')
plt.title('Circularly Symmetric DOE from Radial Profile')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
