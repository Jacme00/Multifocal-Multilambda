# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 23:02:48 2026

@author: zarkm
"""
import numpy as np
from objective import bb_objective
import time
from scipy.optimize import minimize
from simulation import simulate_onaxis
from objective import unpack
import matplotlib.pyplot as plt
import pickle
import config




I_des_raw = np.exp(-0.5 * ((config.z - config.z0)/config.sigma)**2)
I_des1_raw = np.exp(-0.5 * ((config.z - config.z1)/config.sigma1)**2) 
I_des2_raw = np.exp(-0.5 * ((config.z - config.z2)/config.sigma2)**2) #added yet another new lambdas, new sigmas and new desired target intensity profiles
area0_des = np.trapz(I_des_raw, config.z)
area1_des = np.trapz(I_des1_raw, config.z)
area2_des = np.trapz(I_des2_raw, config.z)


# Normalized random coefficients in [0,1]
coeffs_z_norm = np.random.rand(config.n_zernike)    #  Zernike terms
coeffs_f_norm = np.random.rand(config.n_fourier)   #  Fourier terms 350
coeffs_rbf_norm = np.random.rand(config.n_rbf) #  RBF terms


dims      = (len(coeffs_z_norm), len(coeffs_f_norm), len(coeffs_rbf_norm))
total_dim = sum(dims)


def objective(x):
    return bb_objective(
        x, dims, config.r, config.z, config.R,
        config.lambda0, config.lambda1, config.lambda2,
        I_des_raw, I_des1_raw, I_des2_raw,
        area0_des, area1_des, area2_des
    )

# initial guess
x0 = np.concatenate([coeffs_z_norm, coeffs_f_norm, coeffs_rbf_norm])

# callback to print progress
best = {"val": objective(x0)}
counter = {"it": 0}
last_time = time.perf_counter()


t0 = time.perf_counter()
objective(x0)
print("bb_objective time =", time.perf_counter() - t0)


# --- 1) Start both curves at zero ---
I0_best = np.zeros_like(config.z)
I0_tar  = np.zeros_like(config.z)

# --- 3) Tracking state ---
best = {
    'val': objective(x0),   # best‐so‐far loss
    'y'  : I0_best.copy()      # best‐so‐far sim curve
}
counter = {'it': 0}



bounds = [(0.0, 1.0)] * total_dim

res = minimize(
    objective,
    x0,
    method="L-BFGS-B",
    bounds=bounds,
    options={
        "maxiter": 300,
        "maxfun": 200000,
        "disp": True
    }
)


print(res.message)
print("nit:", getattr(res, "nit", None))
print("nfev:", getattr(res, "nfev", None))

plt.ioff()

print("\nPattern search (L-BFGS-B) finished.")
print("  Success:", res.success)
print(f"  Final best MSE = {res.fun:.6e}")



# unpack final
cz_opt, cf_opt, cr_opt = unpack(res.x, dims)
I_opt1 = simulate_onaxis(cz_opt, cf_opt, cr_opt, config.r, config.z, config.R, config.lambda0)
I_opt2 = simulate_onaxis(cz_opt, cf_opt, cr_opt, config.r, config.z, config.R, config.lambda1)
I_opt3 = simulate_onaxis(cz_opt, cf_opt, cr_opt, config.r, config.z, config.R, config.lambda2)
# (You can now plot I_opt vs. the target as before.)


area_sim1 = np.trapezoid(I_opt1, config.z)
area_sim2 = np.trapezoid(I_opt2, config.z)
area_sim3 = np.trapezoid(I_opt3, config.z)

area_des1 = np.trapezoid(I_des_raw, config.z)
area_des2 = np.trapezoid(I_des1_raw, config.z)
area_des3 = np.trapezoid(I_des2_raw, config.z)

I_fin1    = I_des_raw / area_des1
I_fin2    = I_des1_raw / area_des2
I_fin3    = I_des2_raw / area_des3



# 1) Gather everything you actually produced into a dict
vars_to_save = {
    # grids & physical parameters
    'r'        : config.r,
    'Nr'       : config.Nr,
    'z'        : config.z,
    'R'        : config.R,
    'lambda0'  : config.lambda0,
    'lambda1'  : config.lambda1,
    'lambda2'  : config.lambda2,

    # desired target profiles
    'z0'        : config.z0,
    'z1'        : config.z1,
    'z2'        : config.z2,
    'sigma'     : config.sigma,
    'sigma1'    : config.sigma1,
    'sigma2'    : config.sigma2,
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
name='optimization_results3lambdasHR2swapped1.pkl'
with open(name, 'wb') as f:
    pickle.dump(vars_to_save, f)

print("All key variables saved to name")
