"""
Training, evaluation, and uncertainty quantification for ESM-CrossDTA.

Two UQ approaches are implemented:

  MC-Dropout (baseline):
    Run inference n times with dropout active; return mean and std.
    Limitation: std is not guaranteed to correlate with actual error.

  Conformal Prediction (recommended):
    Split conformal regression — calibrate nonconformity scores on a held-out
    calibration set; at inference return prediction intervals with a formal
    marginal coverage guarantee: P(y ∈ interval) ≥ 1−α.
    Unlike MC-dropout, coverage is guaranteed regardless of model architecture.

Reference (conformal):
  Angelopoulos & Bates (2023), "A Gentle Introduction to Conformal Prediction
  and Distribution-Free Uncertainty Quantification," TMLR.
"""

import numpy as np
import torch
import torch.nn as nn


# ── Metrics ──────────────────────────────────────────────────────────────────

def concordance_index(y_true, y_pred):
    """CI: fraction of correctly-ordered pairs. O(n²), suitable for DAVIS scale."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    order  = np.argsort(y_true)
    y_true, y_pred = y_true[order], y_pred[order]
    pairs = correct = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[j] > y_true[i]:
                pairs += 1
                if   y_pred[j] > y_pred[i]: correct += 1
                elif y_pred[j] == y_pred[i]: correct += 0.5
    return correct / pairs if pairs else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn, total = nn.MSELoss(), 0.0
    for drug, prot_emb, prot_mask, y in loader:
        drug, prot_emb, prot_mask, y = (
            drug.to(device), prot_emb.to(device),
            prot_mask.to(device), y.to(device),
        )
        optimizer.zero_grad()
        pred = model(drug, prot_emb, prot_padding_mask=prot_mask)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for drug, prot_emb, prot_mask, y in loader:
        drug, prot_emb, prot_mask = (
            drug.to(device), prot_emb.to(device), prot_mask.to(device)
        )
        preds.append(model(drug, prot_emb, prot_padding_mask=prot_mask).cpu().numpy())
        trues.append(y.numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    return {"mse": float(np.mean((preds - trues) ** 2)),
            "ci":  concordance_index(trues, preds)}


@torch.no_grad()
def predict(model, loader, device):
    """Return (predictions, labels) arrays for conformal calibration."""
    model.eval()
    preds, trues = [], []
    for drug, prot_emb, prot_mask, y in loader:
        drug, prot_emb, prot_mask = (
            drug.to(device), prot_emb.to(device), prot_mask.to(device)
        )
        preds.append(model(drug, prot_emb, prot_padding_mask=prot_mask).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


@torch.no_grad()
def mc_dropout_predict(model, loader, device, n_passes=30):
    """MC-dropout: dropout active during inference for n_passes. Returns mean, std."""
    model.train()   # dropout ON
    runs = []
    for _ in range(n_passes):
        run = []
        for drug, prot_emb, prot_mask, _ in loader:
            drug, prot_emb, prot_mask = (
                drug.to(device), prot_emb.to(device), prot_mask.to(device)
            )
            run.append(model(drug, prot_emb, prot_padding_mask=prot_mask).cpu().numpy())
        runs.append(np.concatenate(run))
    stacked = np.stack(runs)           # (n_passes, n_samples)
    return stacked.mean(0), stacked.std(0)


# ── Conformal Prediction ──────────────────────────────────────────────────────

class ConformalPredictor:
    """
    Split conformal regression.

    Protocol:
      1. Train model on train set.
      2. Reserve a calibration set (10–20% of training pairs, held out from training).
      3. Compute nonconformity scores: s_i = |y_i − ŷ_i| on the calibration set.
      4. At inference: interval = ŷ ± q̂  where
             q̂ = quantile(scores, level=(⌈(n+1)(1−α)⌉/n))
         This gives P(y ∈ interval) ≥ 1−α marginally (finite-sample guarantee).

    Unlike MC-dropout, coverage is guaranteed by construction under the
    exchangeability assumption — no distributional assumptions about the model.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_hat: float = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalPredictor":
        scores  = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        n       = len(scores)
        level   = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = float(np.quantile(scores, min(level, 1.0)))
        return self

    def predict_interval(self, y_pred: np.ndarray):
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_interval().")
        y_pred = np.asarray(y_pred)
        return y_pred - self.q_hat, y_pred + self.q_hat

    def empirical_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        lo, hi = self.predict_interval(y_pred)
        return float(np.mean((np.asarray(y_true) >= lo) & (np.asarray(y_true) <= hi)))

    def interval_width(self) -> float:
        return 2 * self.q_hat if self.q_hat is not None else float("nan")
