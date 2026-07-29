"""
sensitivity_h_reference.py
==========================
Quantifies the effect of an inconsistency found in the source workbook
during verification.

In Sheet1, the Biot number of the second simulation block correctly uses
that block's own coefficient (cell C263, 18,986.53 W/(m2.K)). The final
term of the surface-node equation, however, still references the first
block's coefficient (cell C8, 8,825 W/(m2.K)). The reference was
evidently not updated when the block was duplicated.

The solver reproduces the workbook behaviour by default so that
validation remains meaningful. This script re-runs the second block with
the reference made consistent and reports the difference, establishing
whether the inconsistency is material.

Run after `validate_against_excel.py`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quench_solver import config_part2, solve

OUTDIR = Path(".")


def main() -> None:
    cfg = config_part2()

    as_built = solve(cfg)
    consistent = solve(cfg.with_(h_surface_reference=cfg.h_total))

    surface_as_built = as_built.T_surface[-1]
    surface_consistent = consistent.T_surface[-1]
    difference = surface_consistent - surface_as_built

    print("=" * 64)
    print("SENSITIVITY: surface-node reference coefficient")
    print("=" * 64)
    print(f"  workbook value  h_ref =  8,825.0 W/(m2.K)")
    print(f"  consistent      h_ref = {cfg.h_total:9,.1f} W/(m2.K)")
    print()
    print(f"  surface temperature, as built   : {surface_as_built:8.3f} degC")
    print(f"  surface temperature, consistent : {surface_consistent:8.3f} degC")
    print(f"  difference                      : {difference:+8.3f} degC")
    print(f"  relative                        : "
          f"{abs(difference) / surface_as_built * 100:8.3f} %")
    print()
    print(f"  core temperature, as built      : {as_built.T_core[-1]:8.3f} degC")
    print(f"  core temperature, consistent    : {consistent.T_core[-1]:8.3f} degC")
    print()
    verdict = "immaterial" if abs(difference) < 10 else "material"
    print(f"  assessment                      : {verdict}")

    # -- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(as_built.time, as_built.T_surface, lw=2, color="#d62728",
            label=f"as built (h_ref = 8,825): {surface_as_built:.1f} degC")
    ax.plot(consistent.time, consistent.T_surface, lw=2, ls="--", color="#2ca02c",
            label=f"consistent (h_ref = {cfg.h_total:,.0f}): {surface_consistent:.1f} degC")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Surface temperature [degC]")
    ax.set_title("Effect of the surface-node reference coefficient")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(as_built.time, consistent.T_surface - as_built.T_surface,
            lw=2, color="#7f7f7f")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Difference [degC]")
    ax.set_title(f"Divergence over the quench (final {difference:+.2f} degC)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTDIR / "sensitivity_h_reference.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure written to {OUTDIR / 'sensitivity_h_reference.png'}")


if __name__ == "__main__":
    main()
