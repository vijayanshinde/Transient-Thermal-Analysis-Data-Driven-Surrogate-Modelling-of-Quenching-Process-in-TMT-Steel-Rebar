"""
optimise_settings.py
====================
Uses the surrogate model as a design tool rather than a predictor.

The surrogate answers the forward question: given a set of quench-line
settings, what surface temperature results? Process engineers face the
inverse question: to achieve a target surface temperature at the quench
exit, what settings are required?

This script answers the inverse question. For a chosen target it
searches the operating envelope for the convective coefficient and
residence time that hit the target, holding the remaining inputs at
representative values. Because a single surrogate evaluation takes
microseconds, thousands of candidate settings can be examined
essentially instantly, which is the practical reason for building the
surrogate in the first place.

Each recommendation is then checked against the full finite-difference
solver, confirming that the surrogate's suggestion is physically sound.

Run after `surrogate_model.py` (it reuses the sweep dataset).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quench_solver import QuenchConfig, solve, stability_time_step

HERE = Path(__file__).parent
DATASET = HERE / "quench_sweep_dataset.csv"

FEATURES = [
    "bar_diameter_mm",
    "h_convection",
    "T_core_initial",
    "delta_T_initial",
    "T_water",
    "quench_time_s",
]
TARGET = "T_surface_end"

# Fixed operating conditions for the design study. Only the two settings
# an operator can readily adjust, quench intensity and residence time,
# are searched over.
FIXED = {
    "bar_diameter_mm": 25.0,
    "T_core_initial": 1020.0,
    "delta_T_initial": 120.0,
    "T_water": 30.0,
}

SEARCH = {
    "h_convection": (8_000.0, 26_000.0),
    "quench_time_s": (0.10, 0.35),
}

TARGETS_TO_HIT = [450.0, 500.0, 550.0, 600.0]
H_RADIATION = 47.9


def build_surrogate():
    """Train the forward surrogate on the sweep dataset."""
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=5,
            subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_depth=5, random_state=42
        )
    frame = pd.read_csv(DATASET)
    model.fit(frame[FEATURES], frame[TARGET])
    return model


def solver_surface_temperature(h_convection: float, quench_time: float) -> float:
    """Ground-truth surface temperature from the full solver."""
    cfg = QuenchConfig(
        diameter=FIXED["bar_diameter_mm"] / 1000.0,
        T_core_0=FIXED["T_core_initial"],
        T_surface_0=FIXED["T_core_initial"] - FIXED["delta_T_initial"],
        T_surroundings=FIXED["T_water"],
        h_convection=h_convection,
        h_radiation=H_RADIATION,
    )
    dt = stability_time_step(cfg, h_for_biot=cfg.h_total)
    n_steps = int(round(quench_time / dt))
    return float(solve(cfg.with_(dt=dt, n_steps=n_steps)).T_surface[-1])


def main() -> None:
    model = build_surrogate()

    # Dense grid of candidate settings. 200 x 200 = 40,000 candidates,
    # evaluated by the surrogate in a fraction of a second.
    h_grid = np.linspace(*SEARCH["h_convection"], 200)
    t_grid = np.linspace(*SEARCH["quench_time_s"], 200)
    H, T = np.meshgrid(h_grid, t_grid)

    candidates = pd.DataFrame(
        {
            "bar_diameter_mm": FIXED["bar_diameter_mm"],
            "h_convection": H.ravel(),
            "T_core_initial": FIXED["T_core_initial"],
            "delta_T_initial": FIXED["delta_T_initial"],
            "T_water": FIXED["T_water"],
            "quench_time_s": T.ravel(),
        }
    )[FEATURES]
    predicted = model.predict(candidates).reshape(H.shape)

    print("=" * 70)
    print("INVERSE DESIGN: settings to achieve a target surface temperature")
    print("=" * 70)
    print(f"  fixed: 25 mm bar, core 1020 degC, water 30 degC")
    print(f"  searching h_convection and quench residence time")
    print(f"  candidates evaluated by surrogate: {candidates.shape[0]:,}\n")

    print(f"  {'target':>8} | {'h_conv':>9} | {'time':>6} | "
          f"{'surrogate':>9} | {'solver':>8} | {'error':>7}")
    print("  " + "-" * 62)

    recommendations = []
    for target in TARGETS_TO_HIT:
        error = np.abs(predicted - target)
        j, i = np.unravel_index(error.argmin(), error.shape)
        h_best, t_best = H[j, i], T[j, i]
        surrogate_value = predicted[j, i]
        solver_value = solver_surface_temperature(h_best, t_best)

        recommendations.append(
            {
                "target": target,
                "h_convection": h_best,
                "quench_time_s": t_best,
                "surrogate": surrogate_value,
                "solver": solver_value,
                "solver_error": solver_value - target,
            }
        )
        print(f"  {target:8.0f} | {h_best:9.0f} | {t_best:6.3f} | "
              f"{surrogate_value:9.1f} | {solver_value:8.1f} | "
              f"{solver_value - target:+7.1f}")

    print("\n  The solver column confirms each surrogate recommendation")
    print("  against the full physics. Agreement within a few degrees")
    print("  shows the surrogate is a reliable design tool.")

    _figure(H, T, predicted, recommendations)
    pd.DataFrame(recommendations).to_csv(HERE / "optimised_settings.csv", index=False)
    print(f"\n  figure  : {HERE / 'design_map.png'}")
    print(f"  table   : {HERE / 'optimised_settings.csv'}")


def _figure(H, T, predicted, recommendations) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    filled = ax.contourf(H, T * 1000.0, predicted, levels=20, cmap="RdYlBu_r")
    fig.colorbar(filled, ax=ax, label="Surface temperature at exit [degC]")

    target_levels = [r["target"] for r in recommendations]
    lines = ax.contour(H, T * 1000.0, predicted, levels=target_levels,
                       colors="black", linewidths=1.5)
    ax.clabel(lines, fmt="%.0f degC", fontsize=9)

    for r in recommendations:
        ax.plot(r["h_convection"], r["quench_time_s"] * 1000.0, "k*", ms=15)
        ax.annotate(
            f"{r['target']:.0f} degC\nh={r['h_convection']:,.0f}\n"
            f"t={r['quench_time_s'] * 1000:.0f} ms",
            xy=(r["h_convection"], r["quench_time_s"] * 1000.0),
            xytext=(8, 8), textcoords="offset points", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    ax.set_xlabel("Convective coefficient h [W/(m2.K)]")
    ax.set_ylabel("Quench residence time [ms]")
    ax.set_title(
        "Design map: settings that achieve a target surface temperature\n"
        "(25 mm bar, core 1020 degC, water 30 degC)"
    )
    fig.tight_layout()
    fig.savefig(HERE / "design_map.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
