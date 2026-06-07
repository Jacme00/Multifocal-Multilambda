# Inverse Design of Radial Diffractive Optical Elements

This project explores the inverse design of radially symmetric diffractive optical elements (DOEs) that shape the axial response of light for multiple wavelengths at once.

The main goal is to design a radial DOE profile that causes light from different wavelengths to reach maximum intensity at different prescribed positions along the propagation axis. Instead of starting from a fixed DOE and asking where each wavelength focuses, this project starts from the desired axial behavior and numerically finds the DOE that produces it.

## Overview

The code implements a computational framework for designing circularly symmetric DOEs that can map a set of wavelengths to a corresponding set of desired axial focal locations.

This makes it possible to study and optimize devices that perform multi-wavelength axial beam shaping, where each wavelength is targeted to a different position along the optical axis.

Although the examples in this repository use a specific set of three wavelengths and three target axial positions, the framework is meant to be adaptable to arbitrary user-defined combinations of wavelengths and axial target locations.

## What this project does

* Represents a radial DOE profile using a compact basis expansion
* Simulates its propagation behavior using a scalar diffraction model
* Evaluates the resulting on-axis intensity profile as a function of propagation distance
* Compares the simulated response against user-defined target axial intensity distributions
* Optimizes the DOE coefficients to minimize the mismatch between the simulated and desired multi-wavelength response
* Exports optimized OPD profiles for preliminary full-wave validation in Meep/PyMeep

## Method

The inverse design pipeline is based on three main ingredients:

### 1\. Radial profile parameterization

The DOE is modeled as a radial height or profile function, represented as a combination of:

* radial Zernike-like terms
* Fourier/cosine terms
* Gaussian radial basis functions

This provides a flexible but structured way to describe smooth radial phase profiles.

### 2\. Forward optical simulation

For a candidate DOE, the code computes the axial intensity response for each wavelength using a scalar diffraction / Fresnel-type propagation model. The main quantity of interest is the on-axis intensity as a function of propagation distance.

### 3\. Numerical optimization

The DOE coefficients are optimized using SciPy-based numerical optimization so that the simulated axial response matches a set of desired target profiles. In the current implementation, the objective function compares normalized simulated and target curves for each wavelength.



### 4. Full-wave Meep validation

The project also includes a preliminary full-wave validation workflow using Meep/PyMeep.

The optimized scalar profile is treated as an optical path difference and exported as an equivalent dielectric relief:

```text
OPD(r) = (n - 1) t(r)
t(r) = OPD(r) / (n - 1)
```

Here, `t(r)` is the physical relief height at radius `r`, and `n` is the refractive index used for the lens material.

The exported radial relief is simulated in Meep using cylindrical coordinates, which avoids a full 3D simulation while still solving Maxwell’s equations for the rotationally symmetric structure. The FDTD simulation is restricted to the region around the DOE. A near-to-far monitor is placed after the relief in homogeneous air, so the transmitted electromagnetic field can be propagated to the axial positions of interest without explicitly simulating the full centimeter-scale propagation distance.

Conceptually:

```text
optimized OPD profile
        ↓
dielectric relief profile
        ↓
cylindrical Meep/FDTD simulation near the DOE
        ↓
near-to-far propagation through air
        ↓
scalar-vs-Meep comparison
```

The validation compares the scalar and Meep-predicted on-axis intensity profiles for each wavelength. Current tests show good agreement at the designed focal regions, with convergence checks performed by increasing the Meep spatial resolution.  


## Results

The current optimized DOE was tested for three wavelengths:

```text
lambda0 = 633 nm at z = 0.0699 m
lambda1 = 445 nm at z = 0.1097 m
lambda2 = 532 nm at z = 0.1701 m
```

The scalar model predicts wavelength-dependent axial focusing, with each wavelength targeted to a different propagation distance. The optimized profile was also exported to Meep and validated as a physical dielectric relief using cylindrical FDTD and near-to-far propagation.

Representative Meep validation outputs are stored in:

```text
Results/meep_validation/
```

The validation includes scalar-vs-Meep axial comparisons at 16 and 24 pixels/µm. The main designed focal regions are reproduced by the Meep simulations for all three wavelengths. The 633 nm case shows stable agreement between 16 and 24 pixels/µm, indicating convergence of the main axial focusing behavior.

The full-range axial plots also show additional near-DOE peaks and secondary axial structure. These features are expected in diffractive optical elements, especially when visualizing the complete axial response rather than only the designed focal regions. The main validation result is that the dominant designed focal regions predicted by the scalar model are also reproduced by the full-wave Meep simulations.

## Scientific motivation

Radial DOEs are attractive because they are compact, symmetric, and well suited to applications where the goal is to control how light redistributes along the optical axis.

This project focuses on the inverse problem:

> Given a set of wavelengths and desired axial focal positions, can we compute a single radial DOE that produces that behavior?

That makes the code relevant to problems in:

* computational optics
* diffractive photonics
* beam shaping
* wavelength-dependent axial focusing
* inverse design of optical components

## Current scope

The current code is a research-oriented prototype centered on:

* multi-wavelength axial focusing
* radial symmetry
* scalar diffraction modeling
* numerical optimization of DOE profiles
* preliminary full-wave validation using cylindrical Meep/FDTD simulations

The present examples use Gaussian target profiles centered at chosen axial locations, but the same framework can be extended to other target intensity distributions.

## Repository structure

The layout for this project is:

```text
.
├── README.md
├── basis.py
├── config.py
├── objective.py
├── optimize.py
├── plotting.py
├── simulation.py
├── export_meep_profile.py
├── meep_validate_lens_1wl.py
├── meep_lens_profile.csv
├── meep_lens_metadata.npz
├── optimization_results3lambdasHR2swapped1juny2.pkl
└── Results/
    ├── DOE1.png
    ├── DOE1_raw.png
    ├── DOE1profile.png
    ├── DOE1swapped_raw.png
    ├── simDOE1.png
    ├── sim_swapped.png
    ├── optimization_results3lambdasHR2.pkl
    ├── optimization_results3lambdasHR2swapped.pkl
    └── meep_validation/
        ├── meep_axial_lambda0_res16.png
        ├── meep_axial_lambda1_res16.png
        ├── meep_axial_lambda2_res16.png
        ├── meep_axial_lambda0_res24.png
        ├── meep_axial_lambda1_res24.png
        ├── meep_axial_lambda2_res24.png
        ├── meep_axial_lambda0_res16.npz
        ├── meep_axial_lambda1_res16.npz
        ├── meep_axial_lambda2_res16.npz
        ├── meep_axial_lambda0_res24.npz
        ├── meep_axial_lambda1_res24.npz
        └── meep_axial_lambda2_res24.npz
```

