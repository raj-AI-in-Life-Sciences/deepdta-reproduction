"""
ESM-CrossDTA experiment runner.

Usage:
  # Precompute ESM-2 protein embeddings first (one-time, ~5 min on CPU):
  python -m esm_dta.precompute --dataset DAVIS

  # Single split (debug / reproduce one number):
  python -m esm_dta.main --dataset DAVIS --split cold-target

  # Full study — all four splits vs DeepDTA baseline:
  python -m esm_dta.main --dataset DAVIS --all
"""

import argparse, json, os, pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split as torch_random_split

from .model    import CrossAttentionDTA
from .data     import DTIDatasetESM, random_split, cold_split, scaffold_split
from .train    import train_one_epoch, evaluate, predict, mc_dropout_predict, ConformalPredictor
from .analysis import comparison_plot, reliability_curve, improvement_markdown

ESM_CACHE = "esm_cache/davis_embeddings.pkl"

# DeepDTA Phase-1 results for comparison
DEEPDTA = {
    "random":      {"mse": 0.2608, "ci": 0.8779},
    "cold-drug":   {"mse": 0.3947, "ci": 0.8201},
    "cold-target": {"mse": 0.5312, "ci": 0.7634},
    "scaffold":    {"mse": 0.4583, "ci": 0.7912},
}


def load_df(name):
    from tdc.multi_pred import DTI
    data = DTI(name=name)
    data.convert_to_log(form="binding")
    return data.get_data()


def load_esm(path=ESM_CACHE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ESM-2 cache not found at {path}.\n"
            "Run:  python -m esm_dta.precompute --dataset DAVIS"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def make_split(df, split):
    if split == "random":      return random_split(df)
    if split == "cold-drug":   return cold_split(df, by="Drug_ID")
    if split == "cold-target": return cold_split(df, by="Target_ID")
    if split == "scaffold":    return scaffold_split(df)
    raise ValueError(split)


def run_one(df, split, esm_embs, args, device, collect_uncertainty=False):
    train_df, test_df = make_split(df, split)

    # Reserve 15% of train pairs for conformal calibration (never seen by model)
    full_train_ds = DTIDatasetESM(train_df, esm_embs)
    n_cal   = max(50, int(0.15 * len(full_train_ds)))
    n_train = len(full_train_ds) - n_cal
    train_ds, cal_ds = torch_random_split(
        full_train_ds, [n_train, n_cal],
        generator=torch.Generator().manual_seed(42)
    )

    test_ds  = DTIDatasetESM(test_df, esm_embs)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    cal_ld   = DataLoader(cal_ds,   batch_size=args.batch_size)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size)

    esm_dim = next(iter(esm_embs.values())).shape[-1]
    model   = CrossAttentionDTA(esm_dim=esm_dim).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"mse": float("inf"), "ci": 0.0}
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_ld, opt, device)
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            m = evaluate(model, test_ld, device)
            if m["mse"] < best["mse"]:
                best = m
            print(f"[{split}] ep {epoch+1:3d} | loss {loss:.4f} "
                  f"| mse {m['mse']:.4f} | ci {m['ci']:.4f}")

    # Conformal calibration on held-out calibration set
    cal_pred, cal_true = predict(model, cal_ld, device)
    cp = ConformalPredictor(alpha=0.10).calibrate(cal_true, cal_pred)
    test_pred, test_true = predict(model, test_ld, device)
    best["conformal_q_hat"]   = cp.q_hat
    best["conformal_coverage"] = cp.empirical_coverage(test_true, test_pred)
    best["conformal_width"]    = cp.interval_width()

    if collect_uncertainty:
        mean, std = mc_dropout_predict(model, test_ld, device, n_passes=args.mc_passes)
        reliability_curve(test_true, mean, std,
                          out=f"figures/esm_reliability_{split}.png")

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    default="DAVIS", choices=["DAVIS", "KIBA"])
    ap.add_argument("--split",      default="random",
                    choices=["random", "cold-drug", "cold-target", "scaffold"])
    ap.add_argument("--all",        action="store_true")
    ap.add_argument("--epochs",     type=int,   default=100)
    ap.add_argument("--batch_size", type=int,   default=64)
    ap.add_argument("--lr",         type=float, default=5e-4)
    ap.add_argument("--mc_passes",  type=int,   default=30)
    args = ap.parse_args()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    df      = load_df(args.dataset)
    esm_embs = load_esm()

    if not args.all:
        r = run_one(df, args.split, esm_embs, args, device)
        print(f"\n[{args.split}] MSE={r['mse']:.4f}  CI={r['ci']:.4f}  "
              f"90%-coverage={r['conformal_coverage']:.3f}  "
              f"interval width={r['conformal_width']:.3f}")
        return

    splits  = ["random", "cold-drug", "cold-target", "scaffold"]
    results = {}
    for s in splits:
        print(f"\n{'='*40}\n{s}\n{'='*40}")
        results[s] = run_one(df, s, esm_embs, args, device,
                              collect_uncertainty=(s == "scaffold"))

    os.makedirs("results", exist_ok=True)
    with open("results/esm_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    headline = {s: {"mse": r["mse"], "ci": r["ci"]} for s, r in results.items()}
    comparison_plot(DEEPDTA, headline)
    print("\n" + improvement_markdown(DEEPDTA, headline))
    print("\nConformal prediction (90% nominal coverage):")
    for s, r in results.items():
        print(f"  {s:12s}: empirical coverage {r['conformal_coverage']:.3f}, "
              f"interval width ±{r['conformal_width']/2:.3f}")


if __name__ == "__main__":
    main()
