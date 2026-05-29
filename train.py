"""
Training loop + evaluation metrics for DeepDTA.

Metrics:
  - MSE: the paper's primary loss/metric.
  - Concordance Index (CI): the paper's ranking metric — probability that a
    randomly chosen higher-affinity pair is ranked above a lower one. This is
    what matters for prioritising compounds, so chemists care about it.
  - Pearson / Spearman: report alongside for completeness.

Your extension adds, on top of point predictions:
  - An uncertainty estimate via Monte Carlo dropout (keep dropout ON at
    inference, run N forward passes, take mean + std). Then check CALIBRATION:
    do the model's high-uncertainty predictions actually have higher error?
    A miscalibrated model that looks confident on novel scaffolds is exactly
    the failure mode worth surfacing for a drug-discovery audience.

Targets to validate against (paper, DAVIS, random split, 5-fold):
    MSE ~= 0.261, CI ~= 0.878
If your reproduction lands near these on the random split, the rebuild is
correct and you've earned the right to report the cold-split numbers.
"""

import numpy as np
import torch
import torch.nn as nn


def concordance_index(y_true, y_pred):
    """CI: fraction of correctly-ordered pairs among all comparable pairs."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pairs = 0
    correct = 0
    # O(n^2): fine for test sets of a few thousand; subsample if larger.
    order = np.argsort(y_true)
    y_true, y_pred = y_true[order], y_pred[order]
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[j] > y_true[i]:
                pairs += 1
                if y_pred[j] > y_pred[i]:
                    correct += 1
                elif y_pred[j] == y_pred[i]:
                    correct += 0.5
    return correct / pairs if pairs else 0.0


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    for drug, prot, y in loader:
        drug, prot, y = drug.to(device), prot.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(drug, prot)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for drug, prot, y in loader:
        drug, prot = drug.to(device), prot.to(device)
        preds.append(model(drug, prot).cpu().numpy())
        trues.append(y.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mse = float(np.mean((preds - trues) ** 2))
    ci = concordance_index(trues, preds)
    return {"mse": mse, "ci": ci}


@torch.no_grad()
def mc_dropout_predict(model, loader, device, n_passes=30):
    """Keep dropout ON to get predictive mean + std per pair (your extension)."""
    model.train()  # train mode -> dropout active
    all_runs = []
    for _ in range(n_passes):
        run = []
        for drug, prot, _ in loader:
            drug, prot = drug.to(device), prot.to(device)
            run.append(model(drug, prot).cpu().numpy())
        all_runs.append(np.concatenate(run))
    stacked = np.stack(all_runs)            # (n_passes, n_samples)
    return stacked.mean(0), stacked.std(0)  # mean prediction, uncertainty
