# -*- coding: utf-8 -*-
"""
Created on Sun May  3 02:41:45 2026

@author: zarkm
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plot
from scipy.special import j0
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

from basis import abstract_profile_normalized, revolve_profile_to_2D
from simulation import simulate_onaxis


def load_results(pickle_path):
    with open(pickle_path, "rb") as f:
        vars_loaded = pickle.load(f)
    print(type(vars_loaded))
    return vars_loaded


def plot_saved_results(pickle_path, Nr_a=1000, Nr_b=2000):
    vars_loaded = load_results(pickle_path)

    r = vars_loaded["r"]
    Nr = vars_loaded["Nr"]
    z = vars_loaded["z"]
    R = vars_loaded["R"]

    lambda0 = vars_loaded["lambda0"]
    lambda1 = vars_loaded["lambda1"]
    lambda2 = vars_loaded["lambda2"]

    z0 = vars_loaded["z0"]
    z1 = vars_loaded["z1"]
    z2 = vars_loaded["z2"]

    sigma = vars_loaded["sigma"]
    sigma1 = vars_loaded["sigma1"]
    sigma2 = vars_loaded["sigma2"]

    I_des0 = vars_loaded["I_des0"]
    I_des1 = vars_loaded["I_des1"]
    I_des2 = vars_loaded["I_des2"]

    cz_opt = vars_loaded["cz_opt"]
    cf_opt = vars_loaded["cf_opt"]
    cr_opt = vars_loaded["cr_opt"]

    I_opt_lambda0 = vars_loaded["I_opt_lambda0"]
    I_opt_lambda1 = vars_loaded["I_opt_lambda1"]
    I_opt_lambda2 = vars_loaded["I_opt_lambda2"]

    final_loss = vars_loaded["final_loss"]

    def eval_Nr(Nr_eval):
        r_eval = np.linspace(0, R, Nr_eval)
        I0 = simulate_onaxis(cz_opt, cf_opt, cr_opt, r_eval, z, R, lambda0)
        I1 = simulate_onaxis(cz_opt, cf_opt, cr_opt, r_eval, z, R, lambda1)
        I2 = simulate_onaxis(cz_opt, cf_opt, cr_opt, r_eval, z, R, lambda2)
        return I0, I1, I2

    # ---- Resolution check plot ----
    I0_a, I1_a, I2_a = eval_Nr(Nr_a)
    I0_b, I1_b, I2_b = eval_Nr(Nr_b)

    plt.figure(figsize=(9, 5))

    plt.plot(z, I0_a, "b-",  label=f"{lambda0*1e9:.0f} nm, Nr={Nr_a}")
    plt.plot(z, I0_b, "b--", label=f"{lambda0*1e9:.0f} nm, Nr={Nr_b}")

    plt.plot(z, I1_a, "g-",  label=f"{lambda1*1e9:.0f} nm, Nr={Nr_a}")
    plt.plot(z, I1_b, "g--", label=f"{lambda1*1e9:.0f} nm, Nr={Nr_b}")

    plt.plot(z, I2_a, "r-",  label=f"{lambda2*1e9:.0f} nm, Nr={Nr_a}")
    plt.plot(z, I2_b, "r--", label=f"{lambda2*1e9:.0f} nm, Nr={Nr_b}")

    plt.xlabel("Propagation distance z (m)")
    plt.ylabel("On-axis intensity (arb. units)")
    plt.title("Resolution check: same design, different Nr")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- Through-focus intensity profile plot ----
    I_opt1 = simulate_onaxis(cz_opt, cf_opt, cr_opt, r, z, R, lambda0)
    I_opt2 = simulate_onaxis(cz_opt, cf_opt, cr_opt, r, z, R, lambda1)
    I_opt3 = simulate_onaxis(cz_opt, cf_opt, cr_opt, r, z, R, lambda2)

    area_sim1 = np.trapezoid(I_opt1, z)
    area_sim2 = np.trapezoid(I_opt2, z)
    area_sim3 = np.trapezoid(I_opt3, z)

    area_des1 = np.trapezoid(I_des0, z)
    area_des2 = np.trapezoid(I_des1, z)
    area_des3 = np.trapezoid(I_des2, z)

    I_fin1 = I_des0 / area_des1
    I_fin2 = I_des1 / area_des2
    I_fin3 = I_des2 / area_des3

    plt.figure(figsize=(8, 4))

    plt.plot(z, I_opt1 / area_sim1, color="red",   label=f"Optimized {lambda0*1e9:.0f} nm")
    plt.plot(z, I_opt2 / area_sim2, color="blue",  label=f"Optimized {lambda1*1e9:.0f} nm")
    plt.plot(z, I_opt3 / area_sim3, color="green", label=f"Optimized {lambda2*1e9:.0f} nm")

    plt.plot(z, I_fin1, "--", lw=2, color="red",   label=f"Target {lambda0*1e9:.0f} nm")
    plt.plot(z, I_fin2, "--", lw=2, color="blue",  label=f"Target {lambda1*1e9:.0f} nm")
    plt.plot(z, I_fin3, "--", lw=2, color="green", label=f"Target {lambda2*1e9:.0f} nm")

    plt.xlabel("Propagation distance z (m)")
    plt.ylabel("Intensity (arb. units)")
    plt.title("Through-Focus Intensity Profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- Radial profile plot ----
    h = abstract_profile_normalized(r, cz_opt, cf_opt, cr_opt, R)

    plt.figure(figsize=(6, 4))
    plt.plot(r, h, color="darkorange")
    plt.xlabel(" Radius r")
    plt.ylabel("Height h(r)")
    plt.title("Profile from Coeffs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- 2D DOE map ----
    h2D = revolve_profile_to_2D(h, R, Nr)

    plt.figure(figsize=(5, 5))
    plt.imshow(h2D, extent=[-R, R, -R, R], origin="lower")
    plt.colorbar(label="Height (arb. units)")
    plt.title("Circularly Symmetric DOE from Radial Profile")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


def plot_3d_propagation(coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm,
                        R=4e-3, Nr=200, lambda0=633e-9, z=None):
    if z is None:
        z = np.linspace(0.03, 0.21, 150)

    r = np.linspace(0, R, Nr)
    h = abstract_profile_normalized(r / R, coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm, 1.0)
    phi = 2 * np.pi * h

    k = 2 * np.pi / lambda0
    Nz = len(z)

    I = np.zeros((Nz, Nr))

    for iz, zi in enumerate(z):
        prefac = np.exp(1j * k * zi) / (1j * lambda0 * zi)
        Q = np.exp(1j * k * r**2 / (2 * zi))
        for ir, ri in enumerate(r):
            kernel = j0(k * ri * r / zi)
            integrand = np.exp(1j * phi) * Q * kernel * r
            E_r = prefac * np.trapezoid(integrand, r)
            I[iz, ir] = np.abs(E_r)**2

    R_mesh, Z_mesh = np.meshgrid(r * 1e3, z)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(R_mesh, Z_mesh, I, cmap="viridis", linewidth=0, antialiased=True)

    ax.set_xlabel("Radial distance r (mm)")
    ax.set_ylabel("Propagation distance z (m)")
    ax.set_zlabel("Intensity (arb. units)")
    ax.set_title("3D Propagation of Custom DOE (KF)")
    plt.tight_layout()
    plt.show()


def plot_integrated_focus_profile(coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm,
                                  R=4e-3, Nr=200, lambda0=633e-9, z=None,
                                  r_cut=0.1e-3, z0=0.07, sigma=1e-3):
    if z is None:
        z = np.linspace(0.03, 0.21, 150)

    r = np.linspace(0, R, Nr)
    h = abstract_profile_normalized(r / R, coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm, 1.0)
    phi = 2 * np.pi * h

    k = 2 * np.pi / lambda0
    Nz = len(z)

    I = np.zeros((Nz, Nr))

    for iz, zi in enumerate(z):
        prefac = np.exp(1j * k * zi) / (1j * lambda0 * zi)
        Q = np.exp(1j * k * r**2 / (2 * zi))
        for ir, ri in enumerate(r):
            kernel = j0(k * ri * r / zi)
            integrand = np.exp(1j * phi) * Q * kernel * r
            E_r = prefac * np.trapezoid(integrand, r)
            I[iz, ir] = np.abs(E_r)**2

    idx_cut = np.where(r <= r_cut)[0]
    I_int = np.trapezoid(I[:, :idx_cut[-1] + 1], x=r[:idx_cut[-1] + 1], axis=1)

    window_length = 11
    polyorder = 3

    I_smooth = savgol_filter(I_int, window_length, polyorder)
    I_smooth2 = gaussian_filter1d(I_int, sigma=1.0)

    I_desired_raw = np.exp(-0.5 * ((z - z0) / sigma)**2)

    area_meas = np.trapezoid(I_smooth, z)
    area_desired = np.trapezoid(I_desired_raw, z)
    scale_factor = area_meas / area_desired
    I_desired = I_desired_raw * scale_factor

    plt.figure(figsize=(8, 4))
    plt.plot(z, I_int, label="Raw integrated")
    plt.plot(z, I_smooth2, label="Gaussianly Smoothed", linewidth=2)
    plt.plot(z, I_desired, label="Desired Focus (area-matched)", linestyle="--", linewidth=2)

    plt.xlabel("Propagation distance z (m)")
    plt.ylabel("Integrated intensity")
    plt.title("Through-Focus Intensity Profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()