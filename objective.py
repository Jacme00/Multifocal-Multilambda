# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:44:42 2026

@author: zarkm
"""

from simulation import simulate_onaxis
import numpy as np

def unpack(x, dims):
    n_z, n_f, n_rbf = dims
    cz = x[:n_z]
    cf = x[n_z:n_z + n_f]
    cr = x[n_z + n_f:]
    return cz, cf, cr


def bb_objective(x, dims, r, z, R, lambda0, lambda1, lambda2,
                 I_des_raw, I_des1_raw, I_des2_raw,
                 area0_des, area1_des, area2_des):
    cz, cf, cr = unpack(x, dims)
    
    I0_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda0)
    I1_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda1)
    I2_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda2)
    
    area0_sim = np.trapezoid(I0_sim, z)
    area1_sim = np.trapezoid(I1_sim, z)
    area2_sim = np.trapezoid(I2_sim, z)
    
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


