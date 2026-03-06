import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch

from src.data.dataset import ASLRightHandDataset, collate_fn
from src.model_loader import load_model_from_checkpoint


def _count_parquet(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.parquet")))


def _available_ids(folder: Path):
    ids = set()
    if not folder.exists():
        return ids
    for p in folder.glob("*.parquet"):
        try:
            ids.add(int(p.stem))
        except Exception:
            pass
    return ids


def main():
    p = argparse.ArgumentParser(description="Preflight checks before training/inference")
    p.add_argument("--data_dir", type=str, default="data/asl-fingerspelling")
    p.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to verify compatibility")
    p.add_argument("--max_frames", type=int, default=160)
    p.add_argument("--use_delta_features", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    vocab_json = data_dir / "character_to_prediction_index.json"
    train_landmarks = data_dir / "train_landmarks"
    supplemental_landmarks = data_dir / "supplemental_landmarks"

    print("=== Preflight ===")
    print(f"data_dir: {data_dir.resolve()}")
    print(f"train.csv exists: {train_csv.exists()}")
    print(f"vocab exists: {vocab_json.exists()}")
    print(f"train_landmarks parquet count: {_count_parquet(train_landmarks)}")
    print(f"supplemental_landmarks parquet count: {_count_parquet(supplemental_landmarks)}")

    if not train_csv.exists() or not vocab_json.exists():
        raise SystemExit("Missing required train.csv or vocab json.")

    with open(vocab_json, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"vocab size (raw): {len(vocab)}")

    df = pd.read_csv(train_csv).head(50000).copy()
    if "landmarks_subdir" not in df.columns:
        df["landmarks_subdir"] = "train_landmarks"
    avail_train = _available_ids(train_landmarks)
    avail_supp = _available_ids(supplemental_landmarks)
    df = df[
        ((df["landmarks_subdir"] == "train_landmarks") & (df["file_id"].isin(avail_train)))
        | ((df["landmarks_subdir"] == "supplemental_landmarks") & (df["file_id"].isin(avail_supp)))
    ].copy()
    print(f"rows with available parquet file_id: {len(df)}")
    if "encoded" not in df.columns:
        # lightweight fallback; full encoding logic lives in train.py
        ch2id = {k: int(v) + 1 for k, v in vocab.items()} if "<blank>" not in vocab else {k: int(v) for k, v in vocab.items()}
        df["encoded"] = df["phrase"].astype(str).apply(lambda s: [ch2id[c] for c in s if c in ch2id])

    ds = ASLRightHandDataset(
        df=df,
        landmarks_dir=str(data_dir),
        max_frames=args.max_frames,
        use_delta_features=args.use_delta_features,
        normalize_landmarks=True,
    )
    batch = None
    invalid = 0
    max_scan = min(len(ds), 1000)
    for i in range(max_scan):
        sample = ds[i]
        if sample is None:
            invalid += 1
            continue
        batch = collate_fn([sample])
        break
    if batch is None:
        raise SystemExit(
            f"No valid sample found in first {max_scan} rows (invalid={invalid}). "
            "Check parquet availability, sequence_id coverage and phrase filters."
        )

    X, Y, in_lens, tar_lens = batch
    print(f"sample input shape: {tuple(X.shape)}")
    print(f"sample target length: {int(tar_lens[0])}")
    print(f"effective input length: {int(in_lens[0])}")

    if args.ckpt:
        if not os.path.exists(args.ckpt):
            raise SystemExit(f"Checkpoint not found: {args.ckpt}")
        loaded = load_model_from_checkpoint(args.ckpt, device=torch.device("cpu"))
        print(f"checkpoint input_dim: {loaded.input_dim}")
        print(f"dataset input_dim: {int(X.shape[2])}")
        if int(X.shape[2]) != int(loaded.input_dim):
            raise SystemExit(
                "Input dim mismatch between data pipeline and checkpoint. "
                "Adjust --use_delta_features / preprocessing."
            )
        print("checkpoint compatibility: OK")

    print("Preflight status: OK")


if __name__ == "__main__":
    main()
