import argparse
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import meep as mp
from pathlib import Path

def norm_max(I):
    I = np.asarray(I, dtype=float)
    return I / (np.max(I) + 1e-300)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--which",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0=lambda0, 1=lambda1, 2=lambda2"
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=2.0,
        help="Meep resolution in pixels per micron"
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=60,
        help="Number of axial z points to evaluate"
    )

    parser.add_argument("--n-lens", type=float, default=1.5)
    parser.add_argument("--pml", type=float, default=2.0)
    parser.add_argument("--substrate", type=float, default=2.0)
    parser.add_argument("--padding", type=float, default=4.0)
    parser.add_argument("--after-sources", type=float, default=80.0)

    parser.add_argument(
        "--flip",
        action="store_true",
        help="Use inverted relief: thickness -> max(thickness)-thickness"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load exported lens profile
    # ------------------------------------------------------------

    profile = np.loadtxt(
        "meep_lens_profile.csv",
        delimiter=",",
        skiprows=1
    )

    r_um = profile[:, 0]
    thickness_um = profile[:, 2]

    if args.flip:
        thickness_um = np.max(thickness_um) - thickness_um

    R_um = float(np.max(r_um))
    hmax_um = float(np.max(thickness_um))

    meta = np.load("meep_lens_metadata.npz")

    wavelengths_um = [
        float(meta["lambda0_um"]),
        float(meta["lambda1_um"]),
        float(meta["lambda2_um"]),
    ]

    scalar_intensities = [
        np.asarray(meta["I0"], dtype=float),
        np.asarray(meta["I1"], dtype=float),
        np.asarray(meta["I2"], dtype=float),
    ]

    wavelength_um = wavelengths_um[args.which]
    scalar_I_full = scalar_intensities[args.which]
    z_um_full = np.asarray(meta["z_um"], dtype=float)

    # Downsample axial scan for speed
    if args.max_points < len(z_um_full):
        idx = np.linspace(0, len(z_um_full) - 1, args.max_points)
        idx = np.unique(np.round(idx).astype(int))
        z_um = z_um_full[idx]
        scalar_I = scalar_I_full[idx]
    else:
        z_um = z_um_full
        scalar_I = scalar_I_full

    def height_at_r(rr_um):
        if rr_um < 0 or rr_um > R_um:
            return 0.0
        return float(np.interp(rr_um, r_um, thickness_um))

    # ------------------------------------------------------------
    # Meep geometry
    # ------------------------------------------------------------

    pml_um = args.pml
    substrate_um = args.substrate
    padding_um = args.padding

    size_r_um = R_um + padding_um + pml_um
    size_z_um = pml_um + substrate_um + hmax_um + padding_um + pml_um

    zmin_um = -0.5 * size_z_um
    zmax_um = 0.5 * size_z_um

    substrate_top_um = zmin_um + pml_um + substrate_um
    lens_top_um = substrate_top_um + hmax_um

    glass = mp.Medium(index=args.n_lens)
    air = mp.Medium(index=1.0)

    def material_function(v):
        rr = v.x
        zz = v.z

        # Glass substrate below the DOE.
        if 0 <= rr <= R_um and zmin_um <= zz <= substrate_top_um:
            return glass

        # Glass relief profile.
        local_h = height_at_r(rr)

        if 0 <= rr <= R_um and substrate_top_um <= zz <= substrate_top_um + local_h:
            return glass

        return air

    # ------------------------------------------------------------
    # Source
    # ------------------------------------------------------------

    frequency = 1.0 / wavelength_um
    fwidth = 0.10 * frequency

    source_z_um = zmin_um + pml_um + 0.25 * substrate_um

    sources = [
        mp.Source(
            mp.GaussianSource(
                frequency,
                fwidth=fwidth,
                is_integrated=True,
            ),
            component=mp.Er,
            center=mp.Vector3(0.5 * R_um, 0, source_z_um),
            size=mp.Vector3(R_um, 0, 0),
        ),
        mp.Source(
            mp.GaussianSource(
                frequency,
                fwidth=fwidth,
                is_integrated=True,
            ),
            component=mp.Ep,
            center=mp.Vector3(0.5 * R_um, 0, source_z_um),
            size=mp.Vector3(R_um, 0, 0),
            amplitude=-1j,
        ),
    ]

    sim = mp.Simulation(
        cell_size=mp.Vector3(size_r_um, 0, size_z_um),
        boundary_layers=[mp.PML(thickness=pml_um)],
        sources=sources,
        material_function=material_function,
        resolution=args.resolution,
        dimensions=mp.CYLINDRICAL,
        m=-1,
        eps_averaging=True,
    )

    # ------------------------------------------------------------
    # Near-to-far monitor
    # ------------------------------------------------------------

    n2f_z_um = zmax_um - pml_um - 0.5

    n2f = sim.add_near2far(
        frequency,
        0,
        1,
        mp.Near2FarRegion(
            center=mp.Vector3(0.5 * R_um, 0, n2f_z_um),
            size=mp.Vector3(R_um, 0, 0),
        ),
    )

    # ------------------------------------------------------------
    # Run
    # ------------------------------------------------------------

    print()
    print("Meep lens validation")
    print("--------------------")
    print(f"channel: lambda{args.which}")
    print(f"wavelength: {wavelength_um:.6f} um")
    print(f"resolution: {args.resolution} pixels/um")
    print(f"R: {R_um / 1000:.3f} mm")
    print(f"max relief height: {hmax_um:.4f} um")
    print(f"cell size r: {size_r_um:.2f} um")
    print(f"cell size z: {size_z_um:.2f} um")
    print(f"z scan: {z_um.min()/1000:.2f} to {z_um.max()/1000:.2f} mm")
    print(f"number of z points: {len(z_um)}")
    print(f"flipped relief: {args.flip}")
    print()

    t0 = time.perf_counter()

    sim.run(until_after_sources=args.after_sources)

    print()
    print(f"FDTD time: {time.perf_counter() - t0:.1f} s")
    print("Computing far-field axial scan...")
    print()

    meep_I = []

    for k, z_after_lens_um in enumerate(z_um):
        if k % 10 == 0:
            print(f"far-field point {k + 1}/{len(z_um)}")

        point = mp.Vector3(
            0,
            0,
            lens_top_um + z_after_lens_um
        )

        ff = sim.get_farfield(n2f, point)

        intensity = (
            abs(ff[0]) ** 2
            + abs(ff[1]) ** 2
            + abs(ff[2]) ** 2
        )

        meep_I.append(intensity)

    meep_I = np.asarray(meep_I, dtype=float)

    z_m = z_um * 1e-6

    scalar_peak_z_m = z_m[np.argmax(scalar_I)]
    meep_peak_z_m = z_m[np.argmax(meep_I)]

    print()
    print("Peak comparison")
    print("---------------")
    print(f"scalar peak z: {scalar_peak_z_m:.6f} m")
    print(f"Meep peak z:   {meep_peak_z_m:.6f} m")

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    tag = f"lambda{args.which}_res{args.resolution:g}"
    if args.flip:
        tag += "_flipped"
    
    output_dir = Path("Results") / "meep_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / f"meep_axial_{tag}.npz",
        z_m=z_m,
        scalar_I=scalar_I,
        meep_I=meep_I,
        wavelength_um=wavelength_um,
        resolution=args.resolution,
        flip=args.flip,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(z_m, norm_max(scalar_I), "--", label="scalar")
    plt.plot(z_m, norm_max(meep_I), "-", label="Meep")
    plt.xlabel("z after lens [m]")
    plt.ylabel("max-normalized on-axis intensity")
    plt.title(f"Scalar vs Meep, lambda = {wavelength_um * 1000:.0f} nm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    png_name = output_dir / f"meep_axial_{tag}.png"
    plt.savefig(png_name, dpi=180)

    print()
    print(f"\nSaved:")
    print(f"  {output_dir / f'meep_axial_{tag}.npz'}")
    print(f"  {png_name}")


if __name__ == "__main__":
    main()
