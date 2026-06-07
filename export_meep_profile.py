# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 02:06:27 2026

@author: zarkm
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from basis import abstract_profile_normalized

PICKLE_FILE = "optimization_results3lambdasHR2swapped1juny2.pkl"

lambda_ref = 633e-9
n_lens = 1.5

with open(PICKLE_FILE, "rb") as f:
    data = pickle.load(f)

r = data["r"]
R = data["R"]

cz = data["cz_opt"]
cf = data["cf_opt"]
cr = data["cr_opt"]

# Match your simulate_onaxis convention:
# h is OPD in waves at 633 nm.
rho = r / R
h = abstract_profile_normalized(rho, cz, cf, cr, 1.0)

OPD_m = h * lambda_ref
thickness_m = OPD_m / (n_lens - 1)

print("R [mm]:", R * 1e3)
print("h min:", np.min(h))
print("h max:", np.max(h))
print("h peak-to-valley [waves @ 633 nm]:", np.ptp(h))
print("max thickness [um]:", np.max(thickness_m) * 1e6)
print("peak-to-valley thickness [um]:", np.ptp(thickness_m) * 1e6)

np.savetxt(
    "meep_lens_profile.csv",
    np.column_stack([
        r * 1e6,
        h,
        thickness_m * 1e6,
    ]),
    delimiter=",",
    header="r_um,h_waves_633nm,thickness_um",
    comments=""
)

np.savez(
    "meep_lens_metadata.npz",
    R_um=R * 1e6,
    lambda0_um=data["lambda0"] * 1e6,
    lambda1_um=data["lambda1"] * 1e6,
    lambda2_um=data["lambda2"] * 1e6,
    z_um=data["z"] * 1e6,
    I0=data["I_opt_lambda0"],
    I1=data["I_opt_lambda1"],
    I2=data["I_opt_lambda2"],
)

plt.figure(figsize=(8, 4))
plt.plot(r * 1e3, h)
plt.xlabel("radius [mm]")
plt.ylabel("OPD [waves @ 633 nm]")
plt.title("Optimized OPD profile")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(r * 1e3, thickness_m * 1e6)
plt.xlabel("radius [mm]")
plt.ylabel("equivalent relief height [µm]")
plt.title("Meep-ready dielectric relief, n = 1.5")
plt.grid(True)
plt.tight_layout()
plt.show()