# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 04:21:28 2025
Aci li clave encircled energy com a penalització per a l'optimització'
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

# --- reuse your problem setup & bb_objective/unpack ---
R         = 4e-3
Nr        = 200
r         = np.linspace(0, R, Nr)
z         = np.linspace(0.03, 0.21, 150)
lambda0   = 633e-9
z0, sigma = 0.07, 1e-3
I_des_raw = np.exp(-0.5 * ((z - z0)/sigma)**2)

# Normalized random coefficients in [0,1]
coeffs_z_norm = np.random.rand(6)    # 6 Zernike terms
coeffs_f_norm = np.random.rand(70)   # 50 Fourier terms
coeffs_rbf_norm = np.random.rand(7) # 10 RBF terms


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


def simulate_onaxis(cz, cf, cr, r, z, R, lambda0):
    """
    Compute only I(z, r=0) for each z (on-axis intensity).
    """
    # 1) radial profile and phase
    h   = abstract_profile_normalized(r/R, cz, cf, cr, 1.0)
    phi = 2*np.pi * h               # shape (Nr,)

    k = 2*np.pi / lambda0
    Nz = len(z)

    I_onaxis = np.zeros(Nz)
    for iz, zi in enumerate(z):
        prefac = np.exp(1j*k*zi) / (1j * lambda0 * zi)
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

def bb_objective(x):
    cz, cf, cr = unpack(x)
    I_sim = simulate_onaxis(cz, cf, cr, r, z, R, lambda0)
    area_sim = np.trapz(I_sim, z)
    area_des = np.trapz(I_des_raw, z)
    I_tar    = I_des_raw * (area_sim/area_des)
    return np.mean((I_sim - I_tar)**2)

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

# --- 1) Start both curves at zero ---
I0_best = np.zeros_like(z)
I0_tar  = np.zeros_like(z)

# --- 2) Interactive figure ---
plt.ion()
fig, ax = plt.subplots(figsize=(8,4))
fig.show()
fig.canvas.draw()

# Plot two placeholder lines at zero
line_tar,  = ax.plot(z, I0_tar,  'k--', lw=1, label='Target (area‐matched)')
line_best, = ax.plot(z, I0_best, 'C0-',  lw=2, label='Best so far')

ax.set_xlabel('z (m)')
ax.set_ylabel('Intensity (arb.)')
ax.set_title('Through-Focus Optimization')
ax.legend()
ax.grid(True)

# --- 3) Tracking state ---
best = {
    'val': bb_objective(x0),   # best‐so‐far loss
    'y'  : I0_best.copy()      # best‐so‐far sim curve
}
counter = {'it': 0}

def nm_callback(xk):
    counter['it'] += 1

    # only update every 400 iterations
    if counter['it'] % 400 != 0:
        return

    # 1) evaluate current candidate
    val = bb_objective(xk)
    cz, cf, cr = unpack(xk)
    I_sim      = simulate_onaxis(cz, cf, cr, r, z, R, lambda0)

    # 2) build the area‐matched target for this sim
    area_sim = np.trapz(I_sim, z)
    I_tar_cur = I_des_raw * (area_sim / np.trapz(I_des_raw, z))

    # 3) update best‐so‐far if improved
    if val < best['val']:
        best['val'] = val
        best['y']   = I_sim.copy()
        print(f"Iter {counter['it']:4d}: New best MSE = {val:.6e}")

    # 4) redraw both lines
    line_best.set_ydata(best['y'])
    line_tar .set_ydata(I_tar_cur)

    # rescale
    ymax = max(best['y'].max(), I_tar_cur.max())
    ax.set_ylim(0, ymax*1.1)

    fig.canvas.draw_idle()
    plt.pause(0.01)


# 4) Run Nelder–Mead with this callback
res = minimize(
    bb_objective,
    x0,
    method='Nelder-Mead',
    callback=powell_callback,
    options={
        'maxiter': 300000,
        'xatol':   1e-10,
        'fatol':   1e-10,
        'disp':    False
    }
)

plt.ioff()

print("\nPattern search (Powell) finished.")
print("  Success:", res.success)
print(f"  Final best MSE = {res.fun:.6e}")

# unpack final
cz_opt, cf_opt, cr_opt = unpack(res.x)
I_opt = simulate_onaxis(cz_opt, cf_opt, cr_opt, r, z, R, lambda0)
# (You can now plot I_opt vs. the target as before.)


area_sim = np.trapz(I_opt, z)
area_des = np.trapz(I_des_raw, z)
I_fin    = I_des_raw * (area_sim/area_des)

# 4) plot
plt.figure(figsize=(8,4))
plt.plot(z, I_opt,   label='Best')
#plt.plot(z, I_smooth, label='savgol Smoothed', linewidth=2)
#plt.plot(z, I_opt, label='Gaussianly Smoothed', linewidth=2)
plt.plot(z, I_fin, label='Desired Focus (area-matched)', linestyle='--', linewidth=2)

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




# 1) Gather everything you want to save into a dict
vars_to_save = {
    'r'             : r,
    'z'             : z,
    'R'             : R,
    'lambda0'       : lambda0,
    'coeffs_z_norm' : coeffs_z_norm,
    'coeffs_f_norm' : coeffs_f_norm,
    'coeffs_rbf_norm': coeffs_rbf_norm,
    'cz_opt'        : cz_opt,
    'cf_opt'        : cf_opt,
    'cr_opt'        : cr_opt,
    'I_opt'         : I_opt,
    'I_des_raw'     : I_des_raw,
    'I_fin'         : I_fin,
    'final_MSE'     : res.fun if 'res' in globals() else best['val']
}

# 2) Write to disk
name='optimization_results7cm633lambda3.pkl'
with open(name, 'wb') as f:
    pickle.dump(vars_to_save, f)

print("All key variables saved to name")

















# --------- 1) Load the pickle ----------
fname = "optimization_results7cm633lambda1.pkl"
with open(fname, "rb") as f:
    data = pickle.load(f)

# Pull out what we need
r       = data["r"]          # meters, 0..R
z       = data["z"]          # meters
R       = data["R"]          # meters
I_opt   = data["I_opt"]      # saved best on-axis intensity vs z
I_fin   = data["I_fin"]      # saved area-matched target vs z (from your run)

cz_opt  = data["cz_opt"]
cf_opt  = data["cf_opt"]
cr_opt  = data["cr_opt"]


# --- 1) Recompute full I(z,r) for the optimized profile ---
k = 2*np.pi / lambda0
Nr = len(r)
Nz = len(z)

# rebuild the phase map φ(r) from your optimized coeffs
h_opt = abstract_profile_normalized(r/R, cz_opt, cf_opt, cr_opt, 1.0)
phi   = 2*np.pi * h_opt   # shape (Nr,)

I_full = np.zeros((Nz, Nr))

for iz, zi in enumerate(z):
    prefac = np.exp(1j*k*zi) / (1j * lambda0 * zi)
    Q      = np.exp(1j * k * r**2 / (2*zi))
    for ir, ri in enumerate(r):
        kernel    = j0(k * ri * r / zi)      # Hankel kernel
        integrand = np.exp(1j*phi) * Q * kernel * r
        E_r       = prefac * np.trapz(integrand, r)
        I_full[iz, ir] = np.abs(E_r)**2

# --- 2) 3D surface plot ---
from mpl_toolkits.mplot3d import Axes3D  # ensure this import

R_mesh, Z_mesh = np.meshgrid(r*1e3, z)  # r in mm, z in m

fig = plt.figure(figsize=(10,6))
ax  = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(
    R_mesh, Z_mesh, I_full,
    cmap='viridis', edgecolor='none', antialiased=True
)

ax.set_xlabel('Radial distance $r$ (mm)')
ax.set_ylabel('Propagation distance $z$ (m)')
ax.set_zlabel('Intensity (arb.\ units)')
ax.set_title('3D Propagation of Optimized DOE')
plt.tight_layout()
plt.show()





# 4) plot
plt.figure(figsize=(8,4))
plt.plot(z, I_opt,   label='Best')
#plt.plot(z, I_smooth, label='savgol Smoothed', linewidth=2)
#plt.plot(z, I_opt, label='Gaussianly Smoothed', linewidth=2)
plt.plot(z, I_fin, label='Desired Focus (area-matched)', linestyle='--', linewidth=2)

plt.xlabel('Propagation distance $z$ (m)')
plt.ylabel('Integrated intensity $\int_0^{0.1\,\mathrm{mm}} I(r)\,dr$ (arb. units)')
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


def leakage_fraction_vs_z(phi, r_in, z, lambda0, r_cut=2e-3, r_max=4e-3, Nr_out=120):
    """
    phi: phase on input radial grid r_in (radians), shape (Nr_in,)
    returns L(z) = power outside r_cut divided by total power within r_max
    """
    k = 2*np.pi / lambda0
    r_out = np.linspace(0.0, r_max, Nr_out)
    idx_out = r_out >= r_cut

    # r_in is the integration variable
    L = np.zeros_like(z, dtype=float)

    for iz, zi in enumerate(z):
        prefac = np.exp(1j*k*zi) / (1j*lambda0*zi)
        Q = np.exp(1j * k * r_in**2 / (2*zi))
        g = np.exp(1j*phi) * Q * r_in  # integrand part that depends on r_in

        # Bessel kernel for all output radii at once
        K = j0((k/zi) * np.outer(r_out, r_in))  # (Nr_out, Nr_in)

        E = prefac * np.trapz(K * g, x=r_in, axis=1)  # field at all r_out
        I = np.abs(E)**2

        # power in ring: I(r)*2πr dr  (2π cancels in ratio, but keep r weighting!)
        P_tot = np.trapz(I * r_out, x=r_out)
        P_out = np.trapz(I[idx_out] * r_out[idx_out], x=r_out[idx_out])

        L[iz] = P_out / (P_tot + 1e-30)

    return L
# build phi exactly like your simulation does
h = abstract_profile_normalized(r/R, cz_opt, cf_opt, cr_opt, 1.0)
phi = 2*np.pi * h

Lz = leakage_fraction_vs_z(phi, r, z, lambda0, r_cut=2e-3, r_max=R, Nr_out=120)

L_mean = np.mean(Lz)
L_int  = np.trapz(Lz, z)

print("Mean leakage fraction:", L_mean)
print("Integrated leakage over z:", L_int)