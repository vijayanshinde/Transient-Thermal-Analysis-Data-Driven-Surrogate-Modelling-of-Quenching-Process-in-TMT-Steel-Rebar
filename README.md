# Transient Thermal Analysis and Surrogate Modelling of TMT Rebar Quenching

A validated finite-difference model of the water-quenching stage in
thermo-mechanically treated (TMT) steel reinforcement bar manufacture,
extended with a design-of-experiments study and machine-learning
surrogate models.

Developed during a Research & Development project internship at
**Polaad Steel (Bhagyalaxmi Rolling Mills Pvt. Ltd.)**, in partial
fulfilment of the MSc in Data Analytics and Decision Sciences (DDS),
RWTH Aachen University.

---

## Overview

When a hot steel bar leaves the rolling mill it is quenched in water,
which hardens its surface while the core stays hot. This project models
the transient radial heat conduction during that quench, using an
explicit finite-difference scheme, and then builds a data-driven layer
on top of the validated simulation.

The pipeline has three parts:

1. **Physical model.** An explicit finite-difference solver for 1-D
   transient radial heat conduction in a cooling cylinder, with
   temperature-dependent properties and a combined convective/radiative
   surface boundary. Ported from an existing spreadsheet model and
   validated against it to within `5e-4 °C` across ~9,600 cells.

2. **Design of experiments.** The validated solver is run 600 times
   across the quench-line operating envelope (Latin hypercube sampling)
   to generate a dataset of process settings and outcomes.

3. **Surrogate modelling.** Gradient-boosted regression trees are
   trained on the dataset to predict key outcomes in microseconds, then
   inverted to recommend line settings for a target surface temperature.

---

## Key results

| Item | Result |
|---|---|
| Validation vs spreadsheet | max deviation `5.4e-4 °C` over 9,576 cells |
| Surface exit temperature model | R² = 0.975, MAE = 9.3 °C |
| Peak cooling rate model | R² = 0.983 |
| Rim depth model | R² = 0.723 (mesh-quantisation limited) |
| Inverse design | recommended settings verified within 5 °C of target |
| Solver vs surrogate speed | ~16 ms vs ~microseconds per evaluation |

---

## Repository structure

```
.
├── quench_solver.py             # Core finite-difference solver (import this)
├── validate_against_excel.py    # Cell-by-cell validation vs the spreadsheet
├── parameter_sweep.py           # 600-run design-of-experiments -> dataset
├── surrogate_model.py           # Trains & evaluates the surrogate models
├── analysis_figures.py          # Physics-interpretation & process-map figures
├── sensitivity_h_reference.py   # Sensitivity check on a coefficient reference
├── optimise_settings.py         # Inverts the surrogate into a design tool
├── report_diagrams.py           # Schematic diagrams for the report
│
├── quench_sweep_dataset.csv     # Generated dataset (output of the sweep)
├── surrogate_model_summary.csv  # Model performance summary (output)
├── optimised_settings.csv       # Inverse-design recommendations (output)
│
└── Transient_Explicit_FEA_Fundamental_25mm_bar_example.xlsx
                                  # Source spreadsheet model (place here)
```

> **Note.** The source spreadsheet is required to run the validation
> step. Place it in the repository root with the exact filename above.

---

## Requirements

- Python 3.9 or later
- Required packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl
```

- Optional (recommended) for the surrogate step:

```bash
pip install xgboost shap
```

The surrogate scripts automatically use **XGBoost + SHAP** when
installed, and fall back to **scikit-learn HistGradientBoosting +
permutation importance** otherwise, so the pipeline runs in any
environment.

---

## How to run

Run the scripts from the repository root, in this order. Each stage
depends on the output of the previous one where indicated.

```bash
# 1. Sanity-check the solver (prints endpoint temperatures)
python quench_solver.py

# 2. Validate the Python solver against the source spreadsheet
#    (requires the .xlsx file in the repo root)
python validate_against_excel.py

# 3. Generate the dataset (writes quench_sweep_dataset.csv)
python parameter_sweep.py

# 4. Train and evaluate the surrogate models
#    (reads the dataset, writes surrogate_model_summary.csv + figure)
python surrogate_model.py

# 5. Produce the physics and process-map figures
python analysis_figures.py

# 6. (Optional) Sensitivity check on the coefficient reference
python sensitivity_h_reference.py

# 7. (Optional) Invert the surrogate into a design tool
#    (writes optimised_settings.csv + design_map.png)
python optimise_settings.py

# 8. (Optional) Regenerate the schematic report diagrams
python report_diagrams.py
```

### Dependency between stages

```
quench_solver.py  ──►  validate_against_excel.py   (needs .xlsx)
        │
        └──►  parameter_sweep.py  ──►  quench_sweep_dataset.csv
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        ▼                      ▼                     ▼
               surrogate_model.py     analysis_figures.py   optimise_settings.py
```

---

## Module usage

`quench_solver.py` is importable and is the entry point for any custom
analysis:

```python
from quench_solver import solve, config_part2

result = solve(config_part2())
print(result.T_surface[-1])   # surface temperature at quench exit
print(result.summary())       # headline scalars for the run
```

To run a custom operating point:

```python
from quench_solver import QuenchConfig, solve, stability_time_step

cfg = QuenchConfig(diameter=0.020, h_convection=15000.0, h_radiation=48.0)
dt  = stability_time_step(cfg, h_for_biot=cfg.h_total)
result = solve(cfg.with_(dt=dt, n_steps=int(0.25 / dt)))
```

---

## Method summary

- **Governing equation:** 1-D transient radial heat conduction,
  `∂²T/∂r² + (1/r)∂T/∂r = (1/α) ∂T/∂t`
- **Scheme:** explicit finite difference, 21 radial nodes, stability
  criterion `Fo(1 + Bi) ≤ 1/2`
- **Boundary:** combined convection + linearised radiation at the surface
- **Properties:** temperature-dependent `k(T)` and `α(T)`, piecewise fits
- **Sampling:** Latin hypercube, 600 runs, 6 process inputs
- **Models:** gradient-boosted regression trees (XGBoost / sklearn)
- **Interpretation:** SHAP or permutation importance

---

## Limitations

- Pure conduction model; latent heat of phase transformation is not
  included.
- Single-phase convection correlations; boiling regimes at high surface
  temperature are not modelled.
- Covers the water-quench stage only; the self-tempering stage is future
  work.
- Rim-depth prediction is limited by the discretisation of the radial
  mesh.

See the accompanying report for a full discussion.

---

## Author

**Vijayan Shinde**
MSc Data Analytics and Decision Sciences, RWTH Aachen University
Project Intern, R&D, Polaad Steel (Bhagyalaxmi Rolling Mills Pvt. Ltd.)

---

## Note on confidentiality

Certain process parameters are proprietary to Polaad Steel. This
repository contains the modelling and analysis framework; specific
production values have been generalised where required.
