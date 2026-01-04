# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 03:07:20 2025

@author: zarkm
"""
from Base import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


# --- User DOE: revolve basis profile to 1D h(r) then phase ---
# Radial basis profile functions assumed defined above; reuse coefficients sampling
R = 4e-3  # DOE radius from Bravo paper (4 mm)
Nr = 200
r = np.linspace(0, R, Nr)

# Generate smooth radial profile (height in units of wavelength)
h = abstract_profile_normalized(r/R, coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm, 1.0)  # r/R normalized

# Scale h to physical height: assume max phase depth = lambda (one cycle)
# So h is in [0,1] wavelengths -> physical optical path = h * lambda
lambda0 = 633e-9
phi = 2 * np.pi * h  # phase map in radians

# --- Fresnel propagation parameters ---
k = 2 * np.pi / lambda0
z = np.linspace(0.03, 0.21, 150)  # From 3 cm to 21 cm
Nz = len(z)

# Initialize intensity array
I = np.zeros((Nz, Nr))

# Compute on-axis propagation using Hankel transform for each z and r
for iz, zi in enumerate(z):
    prefac = np.exp(1j * k * zi) / (1j * lambda0 * zi)
    Q = np.exp(1j * k * r**2 / (2 * zi))
    for ir, ri in enumerate(r):
        kernel = j0(k * ri * r / zi)
        integrand = np.exp(1j * phi) * Q * kernel * r
        E_r = prefac * np.trapz(integrand, r)
        I[iz, ir] = np.abs(E_r)**2

# Mesh for plotting
R_mesh, Z_mesh = np.meshgrid(r*1e3, z)  # r in mm, z in m

# 3D visualization
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(R_mesh, Z_mesh, I, cmap='viridis', linewidth=0, antialiased=True)

ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel('Propagation distance z (m)')
ax.set_zlabel('Intensity (arb. units)')
ax.set_title('3D Propagation of Custom DOE (KF)')
plt.tight_layout()
plt.show()






# 1) define radial cutoff and get indices
r_cut = 0.1e-3                     # 0.1 mm in meters
idx_cut = np.where(r <= r_cut)[0]  # indices where r ≤ 0.1 mm

# 2) integrate I(z,r) over r ∈ [0, 0.1 mm] for each z
I_int = np.trapz(I[:, :idx_cut[-1]+1], x=r[:idx_cut[-1]+1], axis=1)

# 3) smooth the through‐focus curve
#    window_length must be odd and ≤ len(z); polyorder < window_length
window_length = 11
polyorder     = 3

I_smooth = savgol_filter(I_int, window_length, polyorder)
# 3a) Gentle smoothing with a small Gaussian filter
sigma = 1.0  # adjust σ for more or less smoothing (try 0.5–2.0)
I_smooth2 = gaussian_filter1d(I_int, sigma=sigma)


# 5) define unscaled target Gaussian profile
z0    = 0.07       # focus position 7 cm in meters
sigma = 1e-3       # standard deviation 1 mm in meters
I_desired_raw = np.exp(-0.5 * ((z - z0) / sigma)**2)

# 6) scale so that the area under I_desired matches that under I_smooth
area_meas    = np.trapz(I_smooth, z)
area_desired = np.trapz(I_desired_raw, z)
scale_factor = area_meas / area_desired
I_desired = I_desired_raw * scale_factor




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
















