"""
quench_solver.py
================
Python port of the transient explicit finite-difference quench model
implemented in `Transient_Explicit_FEA_Fundamental_25mm_bar_example.xlsx`
(Sheet1).

The model solves 1-D radial transient heat conduction in a solid steel
cylinder cooled at its surface by forced convection plus radiation:

        d2T/dr2 + (1/r) dT/dr = (1/alpha) dT/dt

discretised with an explicit (forward-Euler) scheme on 21 radial nodes.

This module reproduces the spreadsheet cell-for-cell, including its
sequential (core -> surface) update sweep, so that results can be
validated against the original workbook before the model is used for
parameter studies.

Author: Vijayan Shinde
Polaad Steel (Bhagyalaxmi Rolling Mills Pvt. Ltd.) - R&D
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Material property correlations (piecewise fits used by the spreadsheet)
# ----------------------------------------------------------------------

def thermal_conductivity(T: float) -> float:
    """Thermal conductivity of steel, k [W/(m.K)], as a function of T [degC].

    Piecewise fit taken from the spreadsheet. The break points at 650 and
    700 degC bracket the ferrite -> austenite transition, where the
    property changes character.
    """
    if T < 650.0:
        return 53.495 - 0.029 * T
    if T < 700.0:
        return 6e-17 * T ** 2 - 0.2008 * T + 164.43
    return 0.0115 * T + 15.889


def thermal_diffusivity(T: float) -> float:
    """Thermal diffusivity of steel, alpha [m2/s], as a function of T [degC].

    Piecewise fit taken from the spreadsheet. Note the spreadsheet applies
    the high-temperature branch for T >= 650 degC.
    """
    if T >= 650.0:
        return (2e-5 * T + 0.035) / 1.0e4
    return (
        -3e-13 * T ** 4
        + 4e-10 * T ** 3
        - 2e-7 * T ** 2
        - 1e-4 * T
        + 0.1409
    ) / 1.0e4


# ----------------------------------------------------------------------
# Model configuration
# ----------------------------------------------------------------------

@dataclass
class QuenchConfig:
    """All inputs required to run a simulation.

    Defaults reproduce Sheet1 of the source workbook.
    """

    # --- geometry -----------------------------------------------------
    diameter: float = 0.025            # bar diameter [m]
    n_intervals: int = 20              # radial intervals, M (=> M+1 nodes)

    # --- initial and boundary temperatures ----------------------------
    T_core_0: float = 1020.0           # initial core temperature [degC]
    T_surface_0: float = 900.0         # initial surface temperature [degC]
    T_surroundings: float = 25.0       # quench medium temperature [degC]

    # --- heat transfer ------------------------------------------------
    h_convection: float = 18938.628601714652   # [W/(m2.K)]
    h_radiation: float = 47.90078983398469     # [W/(m2.K)]

    # --- numerics -----------------------------------------------------
    dt: float = 0.0009776146663452515  # time step [s]
    n_steps: int = 227                 # number of steps after the initial state

    # --- spreadsheet fidelity switches --------------------------------
    # The workbook evaluates the global Fourier number used by the centre
    # node from a *fixed* diffusivity rather than the local one.
    alpha_reference: float = 5.1e-6    # [m2/s]
    # The final term of the surface-node equation references the Part 1
    # convective coefficient in both blocks of the workbook.
    h_surface_reference: float = 8825.0        # [W/(m2.K)]

    # --- derived (populated in __post_init__) -------------------------
    radius: float = field(init=False)
    dx: float = field(init=False)
    n_nodes: int = field(init=False)
    r: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.radius = self.diameter / 2.0
        self.dx = self.radius / self.n_intervals
        self.n_nodes = self.n_intervals + 1
        self.r = np.arange(self.n_nodes) * self.dx

    # -- convenience ---------------------------------------------------

    @property
    def h_total(self) -> float:
        """Combined surface coefficient, h = h_conv + h_rad [W/(m2.K)]."""
        return self.h_convection + self.h_radiation

    @property
    def Fo_reference(self) -> float:
        """Fourier number built from the fixed reference diffusivity."""
        return self.alpha_reference * self.dt / self.dx ** 2

    def with_(self, **kwargs) -> "QuenchConfig":
        """Return a copy of this configuration with fields overridden."""
        return replace(self, **kwargs)


# ----------------------------------------------------------------------
# Results container
# ----------------------------------------------------------------------

@dataclass
class QuenchResult:
    """Output of a simulation run."""

    time: np.ndarray                   # [n_steps+1]
    T: np.ndarray                      # [n_steps+1, n_nodes]  temperatures
    alpha: np.ndarray                  # [n_steps+1, n_nodes]
    k: np.ndarray                      # [n_steps+1, n_nodes]
    Fo: np.ndarray                     # [n_steps+1, n_nodes]
    Bi: np.ndarray                     # [n_steps+1, n_nodes]
    config: QuenchConfig

    # -- convenient views ----------------------------------------------

    @property
    def T_core(self) -> np.ndarray:
        """Temperature history of node 0 (bar centre)."""
        return self.T[:, 0]

    @property
    def T_surface(self) -> np.ndarray:
        """Temperature history of the outermost node (bar surface)."""
        return self.T[:, -1]

    @property
    def cooling_rate_surface(self) -> np.ndarray:
        """Surface cooling rate dT/dt [degC/s] (negative while cooling)."""
        return np.gradient(self.T_surface, self.time)

    def summary(self) -> Dict[str, float]:
        """Headline scalars describing the run."""
        surf, core = self.T_surface, self.T_core
        return {
            "T_surface_start": float(surf[0]),
            "T_surface_end": float(surf[-1]),
            "T_surface_min": float(surf.min()),
            "t_surface_min": float(self.time[int(surf.argmin())]),
            "T_core_start": float(core[0]),
            "T_core_end": float(core[-1]),
            "gap_core_surface_end": float(core[-1] - surf[-1]),
            "gap_subsurface_surface_end": float(self.T[-1, -2] - surf[-1]),
            "peak_surface_cooling_rate": float(self.cooling_rate_surface.min()),
            "Bi_surface_end": float(self.Bi[-1, -1]),
            "Fo_surface_end": float(self.Fo[-1, -1]),
        }


# ----------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------

def initial_profile(cfg: QuenchConfig) -> np.ndarray:
    """Parabolic initial temperature profile across the radius.

        T(r) = (T_core - T_surface) * [1 - (r/R)^2] + T_surface

    This is the profile assumed by the workbook at t = 0. It is a
    modelling assumption representing a bar that has just left the
    rolling mill with its surface already partially cooled.
    """
    return (
        (cfg.T_core_0 - cfg.T_surface_0)
        * (1.0 - (cfg.r / cfg.radius) ** 2)
        + cfg.T_surface_0
    )


def _node_properties(T_node: float, cfg: QuenchConfig) -> Tuple[float, float, float, float]:
    """Return (alpha, k, Fo, Bi) evaluated at a single node temperature."""
    a = thermal_diffusivity(T_node)
    k = thermal_conductivity(T_node)
    Fo = a * cfg.dt / cfg.dx ** 2
    Bi = cfg.h_total * cfg.dx / k
    return a, k, Fo, Bi


def solve(cfg: QuenchConfig | None = None) -> QuenchResult:
    """Run the explicit finite-difference quench simulation.

    The update order within each time step mirrors the spreadsheet's
    left-to-right evaluation:

    1. centre node, using the reference Fourier number;
    2. interior nodes 1..M-1, each using the Fourier number of its
       *inner* neighbour as already updated in this step;
    3. surface node, using the Fourier and Biot numbers of node M-1.

    Neighbour temperatures are always taken from the previous time level.
    """
    cfg = cfg or QuenchConfig()

    n_t, n_x = cfg.n_steps + 1, cfg.n_nodes
    T = np.zeros((n_t, n_x))
    alpha = np.zeros((n_t, n_x))
    k = np.zeros((n_t, n_x))
    Fo = np.zeros((n_t, n_x))
    Bi = np.zeros((n_t, n_x))
    time = np.arange(n_t) * cfg.dt

    # --- initial state ------------------------------------------------
    T[0] = initial_profile(cfg)
    for i in range(n_x):
        alpha[0, i], k[0, i], Fo[0, i], Bi[0, i] = _node_properties(T[0, i], cfg)

    M = cfg.n_intervals
    dx2 = cfg.dx ** 2
    # Constant appearing in the final term of the surface-node equation.
    surface_denominator = cfg.radius / (2.0 * cfg.h_surface_reference)

    # --- march in time ------------------------------------------------
    for n in range(cfg.n_steps):
        Told = T[n]
        Tnew = T[n + 1]

        # (1) centre node: the cylindrical singularity at r = 0 resolves,
        #     via L'Hopital plus symmetry, to a factor of four.
        Tnew[0] = Told[0] + 4.0 * cfg.Fo_reference * (Told[1] - Told[0])
        alpha[n + 1, 0], k[n + 1, 0], Fo[n + 1, 0], Bi[n + 1, 0] = _node_properties(
            Tnew[0], cfg
        )

        # (2) interior nodes
        for i in range(1, M):
            Fo_in = Fo[n + 1, i - 1]          # inner neighbour, this step
            conduction = Fo_in * (Told[i - 1] + Told[i + 1] - 2.0 * Told[i])
            curvature = (Fo_in / (2.0 * i)) * (Told[i + 1] - Told[i - 1])
            Tnew[i] = Told[i] + conduction + curvature
            (
                alpha[n + 1, i],
                k[n + 1, i],
                Fo[n + 1, i],
                Bi[n + 1, i],
            ) = _node_properties(Tnew[i], cfg)

        # (3) surface node: conduction inward, convection and radiation
        #     outward, plus the cylindrical corrections carried by the
        #     workbook's boundary formulation.
        Fo_s = Fo[n + 1, M - 1]
        Bi_s = Bi[n + 1, M - 1]
        k_s = k[n + 1, M - 1]
        Tinf = cfg.T_surroundings

        Tnew[M] = (
            Told[M]
            + 2.0 * Fo_s * (Told[M - 1] - Told[M])
            + 2.0 * Fo_s * Bi_s * (Tinf - Told[M])
            + (Fo_s * Bi_s / M) * (Tinf - Told[M])
            + ((Told[M] - Tinf) * Fo_s * dx2) / (k_s * surface_denominator)
        )
        (
            alpha[n + 1, M],
            k[n + 1, M],
            Fo[n + 1, M],
            Bi[n + 1, M],
        ) = _node_properties(Tnew[M], cfg)

    return QuenchResult(time=time, T=T, alpha=alpha, k=k, Fo=Fo, Bi=Bi, config=cfg)


# ----------------------------------------------------------------------
# Named configurations matching the two blocks of the workbook
# ----------------------------------------------------------------------

def stability_time_step(
    cfg: QuenchConfig,
    h_for_biot: float | None = None,
    safety: float = 0.1 / 6.0,
) -> float:
    """Time step from the explicit-scheme stability criterion.

    Reproduces the workbook expression

        dt = safety * dx^2 / [ alpha (1 + Bi + Bi / (2 M)) ]

    with properties evaluated at the initial temperature of the node at
    one quarter of the radius, which is the reference node used by the
    spreadsheet. The default safety factor of 0.1/6 reproduces the
    workbook's (very conservative) choice, roughly thirty times inside
    the formal limit Fo (1 + Bi) <= 1/2.

    This is required whenever the geometry or the heat transfer
    coefficient is varied, since the admissible step size changes with
    both.
    """
    h = cfg.h_surface_reference if h_for_biot is None else h_for_biot
    reference_node = max(1, cfg.n_intervals // 4)
    r_ref = reference_node * cfg.dx
    T_ref = (
        (cfg.T_core_0 - cfg.T_surface_0) * (1.0 - (r_ref / cfg.radius) ** 2)
        + cfg.T_surface_0
    )
    a = thermal_diffusivity(T_ref)
    k = thermal_conductivity(T_ref)
    Bi = h * cfg.dx / k
    return (
        safety
        * cfg.dx ** 2
        / (a * (1.0 + Bi + Bi / (2.0 * cfg.n_intervals)))
    )


def config_part1() -> QuenchConfig:
    """Sheet1, first simulation block: convection only, h = 8825 W/(m2.K)."""
    return QuenchConfig(h_convection=8825.0, h_radiation=0.0)


def config_part2() -> QuenchConfig:
    """Sheet1, second simulation block: convection plus radiation.

    h = 18938.63 (Churchill-Bernstein) + 47.90 (radiation)
      = 18986.53 W/(m2.K)
    """
    return QuenchConfig()


# ----------------------------------------------------------------------
# Stability
# ----------------------------------------------------------------------

def stability_margin(result: QuenchResult) -> Dict[str, float]:
    """Check the explicit-scheme stability criterion Fo (1 + Bi) <= 1/2.

    Returns the worst-case value encountered anywhere in the solution
    together with the margin to the limit.
    """
    worst = float(np.max(result.Fo * (1.0 + result.Bi)))
    return {
        "worst_Fo_1_plus_Bi": worst,
        "limit": 0.5,
        "margin_factor": 0.5 / worst if worst > 0 else float("inf"),
        "stable": worst <= 0.5,
    }


# ----------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    for label, cfg in (("PART 1  (no radiation)", config_part1()),
                       ("PART 2  (with radiation)", config_part2())):
        res = solve(cfg)
        s = res.summary()
        st = stability_margin(res)
        print(f"\n{label}")
        print(f"  h total                 : {cfg.h_total:,.1f} W/(m2.K)")
        print(f"  surface  {s['T_surface_start']:7.1f} -> {s['T_surface_end']:7.1f} degC")
        print(f"  core     {s['T_core_start']:7.1f} -> {s['T_core_end']:7.1f} degC")
        print(f"  node19 - node20 gap     : {s['gap_subsurface_surface_end']:7.1f} degC")
        print(f"  peak surface cooling    : {s['peak_surface_cooling_rate']:,.0f} degC/s")
        print(f"  Bi surface (final)      : {s['Bi_surface_end']:.6f}")
        print(f"  Fo surface (final)      : {s['Fo_surface_end']:.6f}")
        print(f"  stability Fo(1+Bi)      : {st['worst_Fo_1_plus_Bi']:.4f} "
              f"(limit 0.5, margin x{st['margin_factor']:.0f})")
