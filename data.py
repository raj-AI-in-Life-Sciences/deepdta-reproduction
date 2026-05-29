"""
Data pipeline for DeepDTA reproduction on DAVIS / KIBA.

Two responsibilities:
  1. Label-encode SMILES and protein sequences into fixed-length integer arrays
     (the paper uses character-level integer encoding, NOT one-hot).
  2. Provide the SPLIT strategies. This is where your contribution lives:
     beyond the paper's random split, we add cold-drug and cold-target splits,
     which is the generalization story that separates this repo from a plain
     reproduction.

Data source: the canonical DAVIS/KIBA files are distributed with the original
DeepDTA repo (github.com/hkmztrk/DeepDTA) and mirrored by Therapeutics Data
Commons (TDC). Easiest path: `pip install PyTDC` and pull DAVIS via:

    from tdc.multi_pred import DTI
    data = DTI(name='DAVIS')
    df = data.get_data()   # columns: Drug_ID, Drug, Target_ID, Target, Y

The label transform the paper uses for DAVIS: Y = -log10(Kd / 1e9).
TDC can return this directly with data.convert_to_log(form='binding').
"""

import numpy as np
from torch.utils.data import Dataset

# Character vocabularies (label-encoding). Index 0 is reserved for padding.
CHARISOSMISET = {  # SMILES characters -> int (this is the DeepDTA paper's set)
    "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, ".": 7, "1": 8, "0": 9,
    "3": 10, "2": 11, "5": 12, "4": 13, "7": 14, "6": 15, "9": 16, "8": 17,
    "=": 18, "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24, "F": 25,
    "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, "O": 31, "N": 32, "P": 33,
    "S": 34, "R": 35, "U": 36, "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41,
    "Z": 42, "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, "d": 49,
    "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, "l": 55, "o": 56, "n": 57,
    "s": 58, "r": 59, "u": 60, "t": 61, "y": 62,
}
CHARPROTSET = {  # amino acid characters -> int
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9,
    "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17,
    "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}

MAX_SMILES_LEN = 100   # paper uses 85 for DAVIS; 100 is a safe pad length
MAX_PROT_LEN = 1000    # paper uses 1200 for DAVIS proteins


def label_encode(sequence, charset, max_len):
    """Turn a string into a fixed-length integer vector (0-padded / truncated)."""
    encoded = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(sequence[:max_len]):
        encoded[i] = charset.get(ch, 0)
    return encoded


class DTIDataset(Dataset):
    def __init__(self, df, smiles_col="Drug", prot_col="Target", label_col="Y"):
        self.drugs = [label_encode(s, CHARISOSMISET, MAX_SMILES_LEN) for s in df[smiles_col]]
        self.prots = [label_encode(p, CHARPROTSET, MAX_PROT_LEN) for p in df[prot_col]]
        self.labels = df[label_col].astype(np.float32).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.drugs[idx], self.prots[idx], self.labels[idx]


# ----- SPLIT STRATEGIES (your differentiating contribution) -----

def random_split(df, frac_test=0.2, seed=0):
    """The paper's setting: pairs split at random. Easy, optimistic."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - frac_test))
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]]


def cold_split(df, by="Target_ID", frac_test=0.2, seed=0):
    """
    Cold split: hold out ENTITIES, not pairs. With by='Target_ID', no protein in
    the test set was ever seen in training -> measures generalization to novel
    targets. With by='Drug_ID', novel drugs. This is the realistic, hard setting
    drug-discovery teams actually care about, and where DeepDTA-style models
    typically degrade sharply. Reporting that gap is the SME-flavored finding.
    """
    rng = np.random.default_rng(seed)
    entities = np.array(df[by].unique(), dtype=object)
    rng.shuffle(entities)
    cut = int(len(entities) * (1 - frac_test))
    train_entities = set(entities[:cut])
    train_mask = df[by].isin(train_entities)
    return df[train_mask], df[~train_mask]


def _murcko_scaffold(smiles):
    """Bemis-Murcko scaffold SMILES for a molecule; '' if RDKit can't parse it."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_split(df, smiles_col="Drug", frac_test=0.2, seed=0):
    """
    Scaffold split (the stronger version of a cold-drug split).

    Instead of holding out random drugs, group molecules by their Bemis-Murcko
    scaffold (core ring system) and hold out WHOLE scaffolds. The test set then
    contains chemical cores the model never trained on -- the honest analogue of
    "we designed a new chemotype, will the model still work?". Random or even
    cold-drug splits overstate performance because near-duplicate scaffolds leak
    across the split; scaffold splitting closes that leak.

    Strategy: sort scaffold groups largest-first and greedily fill train until it
    reaches the target size, sending the rest to test. This keeps the split
    deterministic and ensures no scaffold appears on both sides.
    """
    from collections import defaultdict

    scaffolds = defaultdict(list)
    for idx, smi in zip(df.index, df[smiles_col]):
        scaffolds[_murcko_scaffold(smi)].append(idx)

    # Largest scaffold groups first -> train; this is the standard convention
    # (it puts the most common cores in train and rarer ones in test).
    groups = sorted(scaffolds.values(), key=len, reverse=True)

    n_total = len(df)
    n_train_target = int(n_total * (1 - frac_test))
    train_idx, test_idx = [], []
    for group in groups:
        if len(train_idx) + len(group) <= n_train_target:
            train_idx.extend(group)
        else:
            test_idx.extend(group)
    return df.loc[train_idx], df.loc[test_idx]
