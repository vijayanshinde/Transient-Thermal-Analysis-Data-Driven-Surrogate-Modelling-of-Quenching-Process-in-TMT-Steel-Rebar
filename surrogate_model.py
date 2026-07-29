"""
surrogate_model.py
==================
Fits fast surrogate models to the quench simulation dataset.

Motivation
----------
The explicit finite-difference solver is accurate but must march through
thousands of small time steps, which makes it too slow to evaluate
inside an optimisation loop or an online process-control setting. A
surrogate trained on a design-of-experiments sweep reproduces the
solver's input-output behaviour in microseconds, enabling rapid
what-if analysis of quench-line settings.

Models
------
Gradient-boosted regression trees. XGBoost with SHAP attribution is
used when installed; otherwise scikit-learn's histogram gradient
boosting with permutation importance provides an equivalent fallback so
the script runs in any environment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split

OUTDIR = Path(".")
DATASET = OUTDIR / "quench_sweep_dataset.csv"
RANDOM_SEED = 42

FEATURES = [
    "bar_diameter_mm",
    "h_convection",
    "T_core_initial",
    "delta_T_initial",
    "T_water",
    "quench_time_s",
]

TARGETS = {
    "T_surface_end": "Surface temperature at quench exit [degC]",
    "peak_cooling_rate": "Peak surface cooling rate [degC/s]",
    "rim_depth_mm": "Depth cooled below 700 degC [mm]",
}

# --- optional dependencies -------------------------------------------
try:
    from xgboost import XGBRegressor

    HAVE_XGBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor

    HAVE_XGBOOST = False

try:
    import shap

    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False


def build_model():
    """Return an untrained gradient-boosting regressor."""
    if HAVE_XGBOOST:
        return XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_SEED,
    )


def cross_validated_r2(X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    """Five-fold cross-validated coefficient of determination."""
    scores = []
    for train_index, test_index in KFold(
        n_splits=5, shuffle=True, random_state=RANDOM_SEED
    ).split(X):
        model = build_model()
        model.fit(X.iloc[train_index], y.iloc[train_index])
        scores.append(r2_score(y.iloc[test_index], model.predict(X.iloc[test_index])))
    return float(np.mean(scores)), float(np.std(scores))


def importances(model, X_test, y_test) -> pd.Series:
    """Feature attribution, via SHAP when available."""
    if HAVE_SHAP:
        values = shap.Explainer(model)(X_test).values
        return pd.Series(
            np.abs(values).mean(axis=0), index=X_test.columns
        ).sort_values(ascending=False)
    result = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=RANDOM_SEED
    )
    return pd.Series(
        result.importances_mean, index=X_test.columns
    ).sort_values(ascending=False)


def main() -> None:
    frame = pd.read_csv(DATASET)
    X = frame[FEATURES]

    backend = "XGBoost" if HAVE_XGBOOST else "scikit-learn HistGradientBoosting"
    attribution = "SHAP" if HAVE_SHAP else "permutation importance"
    print("=" * 68)
    print("SURROGATE MODEL")
    print("=" * 68)
    print(f"  samples    : {len(frame)}")
    print(f"  features   : {len(FEATURES)}")
    print(f"  regressor  : {backend}")
    print(f"  attribution: {attribution}")

    fig, axes = plt.subplots(2, len(TARGETS), figsize=(5.2 * len(TARGETS), 9))
    summary = []

    for column, (target, description) in enumerate(TARGETS.items()):
        y = frame[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        model = build_model()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        r2 = r2_score(y_test, predicted)
        mae = mean_absolute_error(y_test, predicted)
        cv_mean, cv_std = cross_validated_r2(X, y)
        ranking = importances(model, X_test, y_test)

        summary.append(
            {
                "target": target,
                "R2_holdout": r2,
                "MAE": mae,
                "R2_cv_mean": cv_mean,
                "R2_cv_std": cv_std,
                "top_driver": ranking.index[0],
            }
        )

        print(f"\n{target}")
        print(f"  hold-out R2      : {r2:.4f}")
        print(f"  MAE              : {mae:.3f}")
        print(f"  5-fold CV R2     : {cv_mean:.4f} +/- {cv_std:.4f}")
        print("  drivers          :")
        total = ranking.sum()
        for name, value in ranking.items():
            print(f"      {name:20s} {value / total * 100:5.1f} %")

        # -- parity plot -------------------------------------------------
        ax = axes[0, column]
        ax.scatter(y_test, predicted, s=18, alpha=0.65, edgecolor="none")
        limits = [min(y_test.min(), predicted.min()), max(y_test.max(), predicted.max())]
        ax.plot(limits, limits, "k--", lw=1)
        ax.set_xlabel("Simulated")
        ax.set_ylabel("Surrogate prediction")
        ax.set_title(f"{description}\nR2 = {r2:.4f}, MAE = {mae:.2f}", fontsize=10)
        ax.grid(alpha=0.3)

        # -- importance ranking -----------------------------------------
        ax = axes[1, column]
        ordered = ranking.iloc[::-1]
        ax.barh(range(len(ordered)), ordered.values / total * 100, color="#4C72B0")
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered.index, fontsize=9)
        ax.set_xlabel("Relative importance [%]")
        ax.set_title(f"Drivers of {target}", fontsize=10)
        ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(OUTDIR / "surrogate_model_performance.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(summary).to_csv(OUTDIR / "surrogate_model_summary.csv", index=False)
    print(f"\nFigure  : {OUTDIR / 'surrogate_model_performance.png'}")
    print(f"Summary : {OUTDIR / 'surrogate_model_summary.csv'}")


if __name__ == "__main__":
    main()
