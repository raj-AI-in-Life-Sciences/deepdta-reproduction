"""
Data loading and split strategies for ESM-CrossDTA.

Proteins are represented as per-residue ESM-2 embeddings (pre-computed).
Run `python -m esm_dta.precompute --dataset DAVIS` to generate the cache.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

# ── Vocabularies (same as DeepDTA) ──────────────────────────────────────────

CHARISOSMISET = {
    "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, ".": 7, "1": 8, "0": 9,
    "3": 10, "2": 11, "5": 12, "4": 13, "7": 14, "6": 15, "9": 16, "8": 17,
    "=": 18, "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24, "F": 25,
    "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, "O": 31, "N": 32, "P": 33,
    "S": 34, "R": 35, "U": 36, "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41,
    "Z": 42, "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, "d": 49,
    "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, "l": 55, "o": 56, "n": 57,
    "s": 58, "r": 59, "u": 60, "t": 61, "y": 62,
}

MAX_SMILES_LEN = 100
# ESM-2 hard context limit is 1024 tokens (including [CLS]/[EOS]).
# 512 AA is a safe working limit and covers the vast majority of DAVIS kinases.
MAX_PROT_LEN = 512


def _encode_smiles(smi: str, max_len: int = MAX_SMILES_LEN) -> np.ndarray:
    out = np.zeros(max_len, dtype=np.int64)
    for i, c in enumerate(smi[:max_len]):
        out[i] = CHARISOSMISET.get(c, 0)
    return out


# ── Dataset ──────────────────────────────────────────────────────────────────

class DTIDatasetESM(Dataset):
    """
    Returns four tensors per sample:
      drug_tokens  (MAX_SMILES_LEN,)        int64 — label-encoded SMILES
      prot_emb     (MAX_PROT_LEN, esm_dim)  float32 — padded ESM-2 embeddings
      prot_mask    (MAX_PROT_LEN,)          bool   — True where padding
      label        ()                       float32
    """

    def __init__(
        self,
        df,
        esm_embeddings: dict,         # {target_id: np.ndarray (L, esm_dim)}
        smiles_col: str   = "Drug",
        prot_id_col: str  = "Target_ID",
        label_col: str    = "Y",
        max_prot_len: int = MAX_PROT_LEN,
    ):
        self.drug_tokens = [_encode_smiles(s) for s in df[smiles_col]]
        self.labels      = df[label_col].astype(np.float32).values

        esm_dim = next(iter(esm_embeddings.values())).shape[-1]
        self.prot_embs, self.prot_masks = [], []

        for pid in df[prot_id_col]:
            emb   = esm_embeddings[pid]                   # (L, esm_dim)
            L_use = min(len(emb), max_prot_len)
            pad   = np.zeros((max_prot_len, esm_dim), dtype=np.float32)
            pad[:L_use] = emb[:L_use]
            mask  = np.ones(max_prot_len, dtype=bool)     # True = padding
            mask[:L_use] = False
            self.prot_embs.append(pad)
            self.prot_masks.append(mask)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.drug_tokens[idx], dtype=torch.long),
            torch.tensor(self.prot_embs[idx],   dtype=torch.float),
            torch.tensor(self.prot_masks[idx],  dtype=torch.bool),
            torch.tensor(self.labels[idx],      dtype=torch.float),
        )


# ── ESM-2 embedding computation ──────────────────────────────────────────────

def precompute_esm_embeddings(
    proteins: dict,
    model_name: str = "facebook/esm2_t6_8M_ur50D",
    max_len: int    = MAX_PROT_LEN,
    device: str     = "cpu",
    batch_size: int = 8,
) -> dict:
    """
    Compute per-residue ESM-2 embeddings for a dict of {id: sequence}.

    Uses the ESM-2 8M model (esm2_t6_8M_ur50D) by default — runs on CPU
    in ~5 minutes for the 442 unique proteins in DAVIS.

    Requires: pip install transformers
    Returns:  {id: np.ndarray (min(len(seq), max_len), esm_dim)}
    """
    from transformers import AutoTokenizer, EsmModel

    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = EsmModel.from_pretrained(model_name).to(device).eval()

    ids   = list(proteins.keys())
    seqs  = [proteins[i][:max_len] for i in ids]
    embs  = {}

    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids  = ids[start:start + batch_size]
            batch_seqs = seqs[start:start + batch_size]
            inputs = tokenizer(batch_seqs, return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=max_len + 2,
                               add_special_tokens=True).to(device)
            out = model(**inputs).last_hidden_state   # (B, L+2, d)
            for j, pid in enumerate(batch_ids):
                # strip [CLS] and [EOS]; then remove padding via attention_mask
                mask = inputs["attention_mask"][j]    # 1 = real token
                length = int(mask.sum()) - 2          # exclude CLS + EOS
                embs[pid] = out[j, 1:1 + length, :].cpu().numpy()
            print(f"  {start + len(batch_ids)}/{len(ids)} proteins embedded")

    return embs


# ── Split strategies ──────────────────────────────────────────────────────────

def random_split(df, frac_test=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - frac_test))
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]]


def cold_split(df, by="Target_ID", frac_test=0.2, seed=0):
    rng      = np.random.default_rng(seed)
    entities = np.array(df[by].unique(), dtype=object)
    rng.shuffle(entities)
    cut   = int(len(entities) * (1 - frac_test))
    train = set(entities[:cut])
    mask  = df[by].isin(train)
    return df[mask], df[~mask]


def scaffold_split(df, smiles_col="Drug", frac_test=0.2):
    def murcko(smi):
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            mol = Chem.MolFromSmiles(smi)
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else ""
        except Exception:
            return ""

    groups = defaultdict(list)
    for idx, smi in zip(df.index, df[smiles_col]):
        groups[murcko(smi)].append(idx)

    n_train = int(len(df) * (1 - frac_test))
    train_idx, test_idx = [], []
    for g in sorted(groups.values(), key=len, reverse=True):
        (train_idx if len(train_idx) + len(g) <= n_train else test_idx).extend(g)
    return df.loc[train_idx], df.loc[test_idx]
