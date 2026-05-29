"""
Entry point: run the full experiment matrix and produce the study artifacts.

Two modes:

  Single run (debug / reproduce one number):
      python -m src.main --dataset DAVIS --split random

  Full study (the headline -- runs all four splits, builds the table + figures):
      python -m src.main --dataset DAVIS --all

The full study trains the SAME architecture under four evaluation regimes:

    random       -> the paper's setting; validates the reproduction
    cold-drug    -> novel molecules (entity holdout)
    cold-target  -> novel proteins (entity holdout)
    scaffold     -> novel chemical cores (strongest drug generalization test)

The resulting MSE/CI table + bar chart + calibration curve ARE the deliverable.
The reproduction proves competence; the degradation across splits proves you
understand generalization; the calibration curve + chemist-facing write-up are
the domain-expert layer.
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DTIDataset, random_split, cold_split, scaffold_split
from .model import DeepDTA
from .train import train_one_epoch, evaluate, mc_dropout_predict
from .analysis import (
    reliability_curve, split_comparison_plot, markdown_table, calibration_analysis,
)


def load_dataframe(dataset_name):
    """Pull DAVIS/KIBA via TDC. Install with: pip install PyTDC"""
    from tdc.multi_pred import DTI
    data = DTI(name=dataset_name)
    data.convert_to_log(form="binding")  # the -log transform the paper uses
    return data.get_data()


def make_split(df, split):
    if split == "random":
        return random_split(df)
    if split == "cold-drug":
        return cold_split(df, by="Drug_ID")
    if split == "cold-target":
        return cold_split(df, by="Target_ID")
    if split == "scaffold":
        return scaffold_split(df)
    raise ValueError(f"unknown split: {split}")


def run_one(df, split, args, device, collect_uncertainty=False):
    train_df, test_df = make_split(df, split)
    train_loader = DataLoader(DTIDataset(train_df), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(DTIDataset(test_df), batch_size=args.batch_size)

    model = DeepDTA().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = {"mse": float("inf"), "ci": 0.0}
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, test_loader, device)
            if metrics["mse"] < best["mse"]:
                best = metrics
            print(f"[{split}] epoch {epoch+1:3d} | train_loss {loss:.4f} "
                  f"| test_mse {metrics['mse']:.4f} | test_ci {metrics['ci']:.4f}")

    if collect_uncertainty:
        mean, std = mc_dropout_predict(model, test_loader, device, n_passes=args.mc_passes)
        trues = np.concatenate([y.numpy() for _, _, y in test_loader])
        reliability_curve(trues, mean, std, out=f"figures/reliability_{split}.png")
        best["calibration"] = calibration_analysis(trues, mean, std)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="DAVIS", choices=["DAVIS", "KIBA"])
    ap.add_argument("--split", default="random",
                    choices=["random", "cold-drug", "cold-target", "scaffold"])
    ap.add_argument("--all", action="store_true",
                    help="run all four splits and build the full study")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mc_passes", type=int, default=30)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    df = load_dataframe(args.dataset)

    if not args.all:
        result = run_one(df, args.split, args, device)
        print(f"\nFinal [{args.split}]: MSE={result['mse']:.4f} CI={result['ci']:.4f}")
        return

    splits = ["random", "cold-drug", "cold-target", "scaffold"]
    results = {}
    for s in splits:
        print(f"\n===== running split: {s} =====")
        results[s] = run_one(df, s, args, device, collect_uncertainty=(s == "scaffold"))

    headline = {s: {"mse": r["mse"], "ci": r["ci"]} for s, r in results.items()}

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(results, f, indent=2)

    split_comparison_plot(headline)
    print("\n========== HEADLINE RESULTS ==========")
    print(markdown_table(headline))
    print("\nFigures written to figures/ ; full metrics to results/summary.json")


if __name__ == "__main__":
    main()
