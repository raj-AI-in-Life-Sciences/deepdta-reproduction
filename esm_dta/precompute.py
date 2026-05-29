"""
Precompute ESM-2 per-residue embeddings for all unique proteins in a TDC dataset.

Runtime: ~5 minutes on CPU for the 442 unique kinases in DAVIS.
Model:   ESM-2 8M (esm2_t6_8M_ur50D) — lightweight, runs without GPU.
Output:  esm_cache/davis_embeddings.pkl — dict {target_id: np.ndarray (L, 320)}

Usage:
  python -m esm_dta.precompute --dataset DAVIS
  python -m esm_dta.precompute --dataset KIBA --model facebook/esm2_t33_650M_ur50D
"""

import argparse, os, pickle
from .data import precompute_esm_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="DAVIS")
    ap.add_argument("--model",   default="facebook/esm2_t6_8M_ur50D",
                    help="HuggingFace model ID. esm2_t6_8M_ur50D (320-d) runs on CPU. "
                         "esm2_t33_650M_ur50D (1280-d) needs a GPU.")
    ap.add_argument("--device",  default="cpu")
    ap.add_argument("--out",     default="esm_cache/davis_embeddings.pkl")
    ap.add_argument("--batch",   type=int, default=8)
    args = ap.parse_args()

    from tdc.multi_pred import DTI
    df = DTI(name=args.dataset).get_data()

    proteins = dict(zip(df["Target_ID"], df["Target"]))
    print(f"{len(proteins)} unique proteins in {args.dataset}")

    embs = precompute_esm_embeddings(
        proteins, model_name=args.model,
        device=args.device, batch_size=args.batch,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(embs, f)
    print(f"Saved to {args.out}")
    print(f"Embedding dim: {next(iter(embs.values())).shape[-1]}")


if __name__ == "__main__":
    main()
