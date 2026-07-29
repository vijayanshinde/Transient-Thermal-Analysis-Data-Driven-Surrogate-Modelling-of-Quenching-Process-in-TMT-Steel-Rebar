"""
analysis_figures.py
===================
Produces the physical-interpretation figures for the project report.

Three questions are addressed:

1. What distinguishes the two simulation blocks of Sheet1?
   The convective coefficient more than doubles between them. The
   radiative contribution, often assumed to drive the difference, is
   negligible in water.

2. Why is the surface so much colder than the node beneath it?
   The surface-to-subsurface temperature step is set by the Biot
   number, and the computed gap agrees with the analytical balance.

3. How does quench intensity map onto the process outcome?
   Surface exit temperature and rim depth are traced across the
   sampled operating envelope.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quench_solver import (
    QuenchConfig,
    config_part1,
    config_part2,
    solve,
    thermal_conductivity,
)

OUTDIR = Path(".")

BLUE, RED, GREY = "#1f77b4", "#d62728", "#7f7f7f"


def figure_physics() -> None:
    part1, part2 = solve(config_part1()), solve(config_part2())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # -- (a) cooling histories ------------------------------------------
    ax = axes[0, 0]
    for result, colour, label in (
        (part1, BLUE, "h = 8,825 (Part 1)"),
        (part2, RED, "h = 18,987 (Part 2)"),
    ):
        ax.plot(result.time, result.T_surface, color=colour, lw=2,
                label=f"{label} - surface")
        ax.plot(result.time, result.T_core, color=colour, lw=1.4, ls=":",
                label=f"{label} - core")
    ax.annotate(
        f"{part1.T_surface[-1] - part2.T_surface[-1]:.0f} degC apart",
        xy=(part2.time[-1], 0.5 * (part1.T_surface[-1] + part2.T_surface[-1])),
        xytext=(0.14, 700), fontsize=9,
        arrowprops=dict(arrowstyle="->", color="k", lw=1),
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [degC]")
    ax.set_title("(a) The two blocks differ by convective intensity, not radiation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # -- (b) contributions to h -----------------------------------------
    ax = axes[0, 1]
    cfg2 = config_part2()
    labels = ["Part 1\nconvection", "Part 2\nconvection", "Part 2\nradiation"]
    values = [8825.0, cfg2.h_convection, cfg2.h_radiation]
    bars = ax.bar(labels, values, color=[BLUE, RED, "#ff9896"])
    ax.set_ylabel("h [W/(m2.K)]")
    ax.set_yscale("log")
    ax.set_title("(b) Radiation is 0.25 % of the Part 2 coefficient")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.15,
                f"{value:,.0f}", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # -- (c) surface gap against the Biot balance ------------------------
    ax = axes[1, 0]
    radius_mm = part2.config.r * 1e3
    for step, alpha_value in ((0, 0.35), (60, 0.6), (227, 1.0)):
        ax.plot(radius_mm, part2.T[step], "o-", ms=3, color=RED, alpha=alpha_value,
                label=f"t = {part2.time[step]:.3f} s")
    ax.set_xlabel("Radial position [mm]")
    ax.set_ylabel("Temperature [degC]")

    gap_model = part2.T[-1, -2] - part2.T[-1, -1]
    k_sub = thermal_conductivity(part2.T[-1, -2])
    Bi_effective = part2.config.h_total * part2.config.dx / k_sub
    gap_analytic = Bi_effective * (part2.T[-1, -1] - part2.config.T_surroundings)
    ax.set_title(
        "(c) Surface step matches the Biot balance\n"
        f"model {gap_model:.1f} degC vs analytical {gap_analytic:.1f} degC",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # -- (d) cooling rate across the radius -------------------------------
    ax = axes[1, 1]
    rates = np.gradient(part2.T, part2.time, axis=0)
    ax.plot(radius_mm, rates.min(axis=0), "o-", color=RED, ms=4)
    ax.set_xlabel("Radial position [mm]")
    ax.set_ylabel("Peak cooling rate [degC/s]")
    ax.set_title("(d) Severe cooling is confined to the outer shell")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTDIR / "physics_analysis.png", dpi=150)
    plt.close(fig)


def figure_process_map() -> None:
    frame = pd.read_csv(OUTDIR / "quench_sweep_dataset.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    scatter = ax.scatter(
        frame["h_convection"], frame["T_surface_end"],
        c=frame["quench_time_s"], cmap="viridis", s=16, alpha=0.8,
    )
    ax.set_xlabel("h convection [W/(m2.K)]")
    ax.set_ylabel("Surface temperature at exit [degC]")
    ax.set_title("Quench intensity sets the exit temperature")
    fig.colorbar(scatter, ax=ax, label="Quench time [s]")
    ax.grid(alpha=0.3)

    ax = axes[1]
    scatter = ax.scatter(
        frame["bar_diameter_mm"], frame["peak_cooling_rate"],
        c=frame["h_convection"], cmap="plasma", s=16, alpha=0.8,
    )
    ax.set_xlabel("Bar diameter [mm]")
    ax.set_ylabel("Peak surface cooling rate [degC/s]")
    ax.set_title("Thin bars quench far more severely")
    fig.colorbar(scatter, ax=ax, label="h convection")
    ax.grid(alpha=0.3)

    ax = axes[2]
    scatter = ax.scatter(
        frame["quench_time_s"], frame["rim_depth_mm"],
        c=frame["h_convection"], cmap="plasma", s=16, alpha=0.8,
    )
    ax.set_xlabel("Quench residence time [s]")
    ax.set_ylabel("Depth below 700 degC [mm]")
    ax.set_title("Residence time and intensity set the rim depth")
    fig.colorbar(scatter, ax=ax, label="h convection")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTDIR / "process_map.png", dpi=150)
    plt.close(fig)


def main() -> None:
    figure_physics()
    figure_process_map()
    print(f"Written: {OUTDIR / 'physics_analysis.png'}")
    print(f"Written: {OUTDIR / 'process_map.png'}")


if __name__ == "__main__":
    main()
