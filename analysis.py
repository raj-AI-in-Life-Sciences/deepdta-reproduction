"""
Analysis + figures for the generalization study.

This module turns raw predictions into the artifacts that make the project
read as a real study rather than a code dump:

  1. calibration_analysis: does MC-dropout uncertainty actually track error?
     Bins predictions by predicted uncertainty and reports mean absolute error
     per bin. A well-calibrated model shows error rising monotonically with
     uncertainty. A model that is confidently wrong on novel scaffolds (common
     for DTI models off-distribution) will show flat or non-monotonic error --
     that finding is the point worth writing up for a chemistry audience.

  2. reliability_curve: scatter of |error| vs predicted std, with a binned
     trend line.

  3. split_comparison_plot: bar chart of MSE / CI across the four splits, the
     single figure that tells the whole story.

All plots save to figures/ so they drop straight into the README.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt


def calibration_analysis(y_true, y_pred_mean, y_pred_std, n_bins=10):
    """Bin by predicted uncertainty; report mean abs error per bin."""
    abs_err = np.abs(np.asarray(y_true) - np.asarray(y_pred_mean))
    std = np.asarray(y_pred_std)

    # Equal-count bins on uncertainty (quantile bins handle skewed std nicely).
    order = np.argsort(std)
    bins = np.array_split(order, n_bins)

    rows = []
    for b, idx in enumerate(bins):
        rows.append({
            "bin": b,
            "mean_uncertainty": float(std[idx].mean()),
            "mean_abs_error": float(abs_err[idx].mean()),
            "n": len(idx),
        })
    return rows


def reliability_curve(y_true, y_pred_mean, y_pred_std, out="figures/reliability.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    abs_err = np.abs(np.asarray(y_true) - np.asarray(y_pred_mean))
    std = np.asarray(y_pred_std)

    rows = calibration_analysis(y_true, y_pred_mean, y_pred_std)
    bx = [r["mean_uncertainty"] for r in rows]
    by = [r["mean_abs_error"] for r in rows]

    plt.figure(figsize=(6, 5))
    plt.scatter(std, abs_err, s=6, alpha=0.15, label="per-pair")
    plt.plot(bx, by, "o-", color="crimson", label="binned mean")
    plt.xlabel("Predicted uncertainty (MC-dropout std)")
    plt.ylabel("Absolute error")
    plt.title("Does uncertainty track error?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def split_comparison_plot(results, out="figures/split_comparison.png"):
    """
    results: dict like
        {"random": {"mse":0.26,"ci":0.88}, "cold-target": {...}, ...}
    """
    os.makedirs(os.path.dirname(out), exist_ok=True)
    splits = list(results.keys())
    mse = [results[s]["mse"] for s in splits]
    ci = [results[s]["ci"] for s in splits]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.bar(splits, mse, color="steelblue")
    a1.set_title("MSE by split (lower = better)")
    a1.tick_params(axis="x", rotation=30)
    a2.bar(splits, ci, color="seagreen")
    a2.set_ylim(0.5, 1.0)
    a2.set_title("Concordance Index by split (higher = better)")
    a2.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def markdown_table(results):
    """Render the headline results dict as a Markdown table for the README."""
    lines = ["| Split | MSE | CI |", "|---|---|---|"]
    for split, m in results.items():
        lines.append(f"| {split} | {m['mse']:.4f} | {m['ci']:.4f} |")
    return "\n".join(lines)
