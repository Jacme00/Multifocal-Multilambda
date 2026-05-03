# Inverse Design of Radial Diffractive Optical Elements

This project explores the inverse design of radially symmetric diffractive optical elements (DOEs) that shape the axial response of light for multiple wavelengths at once.

The main goal is to design a radial DOE profile that causes light from different wavelengths to reach maximum intensity at different prescribed positions along the propagation axis. Instead of starting from a fixed DOE and asking where each wavelength focuses, this project starts from the desired axial behavior and numerically finds the DOE that produces it.

## Overview

The code implements a computational framework for designing circularly symmetric DOEs that can map a set of wavelengths to a corresponding set of desired axial focal locations.

This makes it possible to study and optimize devices that perform multi-wavelength axial beam shaping, where each wavelength is targeted to a different position along the optical axis.

Although the examples in this repository use a specific set of three wavelengths and three target axial positions, the framework is meant to be adaptable to arbitrary user-defined combinations of wavelengths and axial target locations.

## What this project does

- Represents a radial DOE profile using a compact basis expansion
- Simulates its propagation behavior using a scalar diffraction model
- Evaluates the resulting on-axis intensity profile as a function of propagation distance
- Compares the simulated response against user-defined target axial intensity distributions
- Optimizes the DOE coefficients to minimize the mismatch between the simulated and desired multi-wavelength response

## Method

The inverse design pipeline is based on three main ingredients:

### 1. Radial profile parameterization

The DOE is modeled as a radial height or profile function, represented as a combination of:

- radial Zernike-like terms
- Fourier/cosine terms
- Gaussian radial basis functions

This provides a flexible but structured way to describe smooth radial phase profiles.

### 2. Forward optical simulation

For a candidate DOE, the code computes the axial intensity response for each wavelength using a scalar diffraction / Fresnel-type propagation model. The main quantity of interest is the on-axis intensity as a function of propagation distance.

### 3. Numerical optimization

The DOE coefficients are optimized using SciPy-based numerical optimization so that the simulated axial response matches a set of desired target profiles. In the current implementation, the objective function compares normalized simulated and target curves for each wavelength.

## Scientific motivation

Radial DOEs are attractive because they are compact, symmetric, and well suited to applications where the goal is to control how light redistributes along the optical axis.

This project focuses on the inverse problem:

> Given a set of wavelengths and desired axial focal positions, can we compute a single radial DOE that produces that behavior?

That makes the code relevant to problems in:

- computational optics
- diffractive photonics
- beam shaping
- wavelength-dependent axial focusing
- inverse design of optical components

## Current scope

The current code is a research-oriented prototype centered on:

- multi-wavelength axial focusing
- radial symmetry
- scalar diffraction modeling
- numerical optimization of DOE profiles

The present examples use Gaussian target profiles centered at chosen axial locations, but the same framework can be extended to other target intensity distributions.

## Repository structure

The layout for this project is:

```text
.
├── README.md
├── basis.py
├── simulation.py
├── objective.py
├── optimize.py
├── plotting.py
├── run_demo.py
├── results/
│   ├── DOE1.png
│   ├── DOE1_raw.png
│   ├── DOE1profile.png
│   ├── DOE1swapped_raw.png
│   ├── simDOE1.png
│   ├── sim_swapped.png
│   ├── optimization_results3lambdasHR2.pkl
│   └── optimization_results3lambdasHR2swapped.pkl
└── archive/
    └── original untidy archives