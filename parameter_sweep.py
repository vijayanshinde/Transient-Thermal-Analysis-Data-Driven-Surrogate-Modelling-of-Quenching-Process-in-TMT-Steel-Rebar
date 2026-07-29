"""
parameter_sweep.py
==================
Generates a design-of-experiments dataset by running the validated
quench solver across the operating envelope of the quench line.

The finite-difference solver is accurate but too slow to sit inside an
optimisation or control loop. This script produces the training data
from which a fast surrogate model is fitted in `surrogate_model.py`.

Process inputs varied
---------------------
    bar_diameter_mm     rolled product size
    h_convection        quench intensity (flow rate, velocity, correlation)
    T_core_initial      mill exit temperature
    delta_T_initial     initial core-to-surface temperature difference
    T_water             quench water temperature
    quench_time_s       residence time in the quench box

Responses recorded
------------------
    T_surface_end       surface temperature leaving the quench box
    T_surface_min       minimum surface temperature reached
    peak_cooling_rate   maximum surface cooling rate
    gap_core_surface    core-to-surface temperature difference at exit
    rim_depth_mm        depth of material cooled below 700 degC
    mean_cooling_rate   mean surface cooling rate over the quench
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from quench_solver import QuenchConfig, solve, stability_time_step

OUTDIR = Path(".")
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SAMPLES = 600

# Sampling ranges. These span a plausible rebar quench-line envelope and
# should be narrowed to the mill's true operating window once confirmed.
RANGES = {
    "bar_diameter_mm": (10.0, 32.0),
    "h_convection": (8_000.0, 26_000.0),
    "T_core_initial": (950.0, 1100.0),
    "delta_T_initial": (60.0, 160.0),
    "T_water": (25.0, 55.0),
    "quench_time_s": (0.10, 0.35),
}

# Austenite decomposition onset. Material cooled below this during the
# quench is capable of forming the hardened rim.
RIM_THRESHOLD_C = 700.0

# Radiative contribution is small in water; held at the workbook value.
H_RADIATION = 47.9


def run_case(sample: dict) -> dict | None:
    """Run one simulation and reduce it to a row of responses."""
    diameter = sample["bar_diameter_mm"] / 1000.0
    T_core = sample["T_core_initial"]
    T_surface = T_core - sample["delta_T_initial"]

    cfg = QuenchConfig(
        diameter=diameter,
        T_core_0=T_core,
        T_surface_0=T_surface,
        T_surroundings=sample["T_water"],
        h_convection=sample["h_convection"],
        h_radiation=H_RADIATION,
    )

    # The admissible time step depends on geometry and heat transfer, so
    # it must be recomputed for every case rather than inherited.
    dt = stability_time_step(cfg, h_for_biot=cfg.h_total)
    n_steps = int(round(sample["quench_time_s"] / dt))
    if not (10 <= n_steps <= 20_000):
        return None

    result = solve(cfg.with_(dt=dt, n_steps=n_steps))

    surface = result.T_surface
    final_profile = result.T[-1]

    # Rim depth: distance from the surface over which the material has
    # been taken below the austenite decomposition threshold.
    below = np.where(final_profile < RIM_THRESHOLD_C)[0]
    rim_depth_mm = (
        (cfg.radius - cfg.r[below.min()]) * 1e3 if below.size else 0.0
    )

    row = dict(sample)
    row.update(
        {
            "T_surface_initial": T_surface,
            "dt": dt,
            "n_steps": n_steps,
            "T_surface_end": float(surface[-1]),
            "T_surface_min": float(surface.min()),
            "peak_cooling_rate": float(result.cooling_rate_surface.min()),
            "mean_cooling_rate": float(
                (surface[-1] - surface[0]) / result.time[-1]
            ),
            "gap_core_surface": float(result.T_core[-1] - surface[-1]),
            "gap_subsurface_surface": float(final_profile[-2] - surface[-1]),
            "T_core_end": float(result.T_core[-1]),
            "rim_depth_mm": float(rim_depth_mm),
            "Bi_surface_end": float(result.Bi[-1, -1]),
            "Fo_surface_end": float(result.Fo[-1, -1]),
            "stability_worst": float(np.max(result.Fo * (1.0 + result.Bi))),
        }
    )
    return row


def main() -> None:
    names = list(RANGES)
    sampler = qmc.LatinHypercube(d=len(names), seed=RANDOM_SEED)
    unit = sampler.random(N_SAMPLES)
    lower = np.array([RANGES[n][0] for n in names])
    upper = np.array([RANGES[n][1] for n in names])
    design = qmc.scale(unit, lower, upper)

    print(f"Running {N_SAMPLES} simulations over {len(names)} process inputs")
    rows, skipped = [], 0
    for j, values in enumerate(design):
        row = run_case(dict(zip(names, values)))
        if row is None:
            skipped += 1
        else:
            rows.append(row)
        if (j + 1) % 100 == 0:
            print(f"  {j + 1:4d}/{N_SAMPLES} complete")

    frame = pd.DataFrame(rows)
    path = OUTDIR / "quench_sweep_dataset.csv"
    frame.to_csv(path, index=False)

    print(f"\nCompleted {len(frame)} runs ({skipped} skipped)")
    print(f"Dataset written to {path}")
    print(f"\nAll runs numerically stable: "
          f"{bool((frame['stability_worst'] <= 0.5).all())} "
          f"(worst Fo(1+Bi) = {frame['stability_worst'].max():.4f})")

    print("\nResponse ranges")
    for column in [
        "T_surface_end",
        "peak_cooling_rate",
        "gap_core_surface",
        "rim_depth_mm",
    ]:
        series = frame[column]
        print(f"  {column:22s} {series.min():10.1f} to {series.max():10.1f}"
              f"   (mean {series.mean():9.1f})")


if __name__ == "__main__":
    main()
