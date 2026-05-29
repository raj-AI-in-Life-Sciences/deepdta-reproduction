# DeepDTA Reproduction + Generalization Study

A clean PyTorch reproduction of [DeepDTA](https://arxiv.org/abs/1801.10193) (Öztürk et al., 2018) extended with a rigorous generalization study across four evaluation regimes. The main finding: performance reported under random splits substantially overstates real-world utility—models degrade sharply on novel scaffolds and unseen targets.

## Why this project

Drug-discovery ML models are routinely benchmarked under random splits, which inflate metrics by allowing near-duplicate drug–target pairs across train and test. This project reproduces the DeepDTA baseline faithfully (to validate the implementation), then evaluates the same model under three progressively harder and more realistic splits. The degradation table is the headline result, and the MC-dropout calibration analysis reveals whether the model's uncertainty estimates are honest on novel inputs.

## Repository structure

```
deepdta-reproduction/
├── src/
│   ├── model.py      # CNNEncoder + DeepDTA (two-tower 1D-CNN, FC head)
│   ├── data.py       # Label encoding, DTIDataset, random/cold/scaffold splits
│   ├── train.py      # Training loop, MSE/CI evaluation, MC-dropout uncertainty
│   ├── analysis.py   # Calibration analysis, reliability curve, split comparison plot
│   └── main.py       # Experiment runner (single split or full study)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

RDKit is required for scaffold splitting. If `pip install rdkit` fails on your platform, use the conda alternative:

```bash
conda install -c conda-forge rdkit
```

## Running the experiments

**Reproduce the paper's random-split number (validation gate):**

```bash
python -m src.main --dataset DAVIS --split random --epochs 100
```

Expected: MSE ≈ 0.261, CI ≈ 0.878 on DAVIS. If your numbers match, the reproduction is correct. Do not proceed to the generalization study until they do.

**Full generalization study (all four splits):**

```bash
python -m src.main --dataset DAVIS --all --epochs 100
```

This trains four separate models (same architecture, different splits), produces `results/summary.json`, and writes figures to `figures/`.

## The four evaluation regimes

| Split | What it holds out | Why it matters |
|---|---|---|
| `random` | Random pairs | Paper's setting; validates reproduction |
| `cold-drug` | Whole drug entities | Novel molecules not seen in training |
| `cold-target` | Whole protein entities | Novel targets not seen in training |
| `scaffold` | Bemis-Murcko scaffold groups | Novel chemical cores (strongest generalization test) |

Scaffold splitting is the most realistic drug-generalization benchmark: it holds out entire chemical scaffold families so the test set contains core ring systems the model never trained on—the honest analogue of "we designed a new chemotype, will the model still work?"

## Expected results (DAVIS, 100 epochs)

| Split | MSE | CI |
|---|---|---|
| random | ~0.261 | ~0.878 |
| cold-drug | higher | lower |
| cold-target | higher | lower |
| scaffold | highest | lowest |

The degradation across splits is the study's main finding. The scaffold and cold-target gaps are the ones a medicinal chemist should care about most—they represent performance on genuinely novel inputs, not statistical replicates of training pairs.

## Uncertainty calibration

MC-dropout uncertainty (30 forward passes with dropout active) is collected for the scaffold split. The reliability curve in `figures/reliability_scaffold.png` shows whether predicted uncertainty tracks actual absolute error. A well-calibrated model has a monotonically increasing trend; flat or non-monotonic curves indicate overconfidence on novel scaffolds—a common failure mode in DTI models.

## Honest caveats

- DAVIS is a kinase-focused dataset with high sequence similarity among targets, so "cold-target" is not perfectly cold—there is structural leakage. The reported cold-target gaps are therefore lower bounds on degradation for truly novel target families.
- The concordance index is computed O(n²); it is fine for DAVIS test sets but will be slow on KIBA. Subsample the test set for CI evaluation if using KIBA.
- The scaffold split uses a greedy largest-first bin-packing strategy, which is the standard convention. It biases common scaffolds toward train and rare scaffolds toward test, which understates test performance relative to a random scaffold assignment.

## Reference

Öztürk, H., Özgür, A., & Ozkirimli, E. (2018). DeepDTA: deep drug–target binding affinity prediction. *Bioinformatics*, 34(17), i821–i829. https://doi.org/10.1093/bioinformatics/bty593
