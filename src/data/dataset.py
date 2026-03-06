import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import pyarrow.parquet as pq

MAX_FRAMES_DEFAULT = 160

_RIGHT_HAND_COLS_BY_PATH: Dict[str, List[str]] = {}


def _get_right_hand_cols(parquet_path: str) -> List[str]:
    """
    Detect columns containing 'right_hand' from a parquet schema (cached globally).
    Kaggle ASL Fingerspelling parquets store flattened landmarks per frame as columns.
    """
    global _RIGHT_HAND_COLS_BY_PATH
    if parquet_path not in _RIGHT_HAND_COLS_BY_PATH:
        pq_file = pq.ParquetFile(parquet_path)
        cols = [c for c in pq_file.schema.names if "right_hand" in c]
        if not cols:
            raise ValueError(
                f"No columns containing 'right_hand' found in parquet schema: {parquet_path}"
            )
        _RIGHT_HAND_COLS_BY_PATH[parquet_path] = cols
    return _RIGHT_HAND_COLS_BY_PATH[parquet_path]


def read_right_hand_sequence(parquet_path: str, sequence_id: int) -> np.ndarray:
    cols = _get_right_hand_cols(parquet_path)
    table = pq.read_table(
        parquet_path,
        filters=[("sequence_id", "=", sequence_id)],
        columns=cols,
    )
    X = table.to_pandas().values.astype(np.float32)  # (T, D)
    return X


def count_valid_frames(X: np.ndarray) -> int:
    # A frame is valid if not all values are NaN
    return int(np.sum(~np.all(np.isnan(X), axis=1)))


def normalize_frames(X: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Pad/truncate to fixed length, keeping NaNs as-is for valid-frame counting.
    Pads with zeros.
    """
    T, D = X.shape
    if T > max_frames:
        return X[:max_frames]
    if T < max_frames:
        pad = np.zeros((max_frames - T, D), dtype=np.float32)
        return np.vstack([X, pad])
    return X


def _center_and_scale_frames(X: np.ndarray, landmark_scale_mode: str = "median_radius") -> np.ndarray:
    """
    Normalize hand landmarks frame-wise:
    - center around wrist (first landmark xyz)
    - scale by median 2D radius to reduce distance-to-camera variance
    """
    T, D = X.shape
    if D % 3 != 0:
        return X

    out = X.copy()
    pts = out.reshape(T, D // 3, 3)
    wrist = pts[:, 0:1, :]
    pts = pts - wrist

    if landmark_scale_mode == "median_radius":
        radii = np.linalg.norm(pts[:, :, :2], axis=2)  # (T, N)
        valid = radii > 1e-6
        scale = np.where(valid, radii, np.nan)
        scale = np.nanmedian(scale, axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
        pts = pts / scale[:, None, None]

    return pts.reshape(T, D).astype(np.float32)


def _append_delta_features(X: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(X, dtype=np.float32)
    if X.shape[0] > 1:
        delta[1:] = X[1:] - X[:-1]
    return np.concatenate([X, delta], axis=1).astype(np.float32)


class ASLRightHandDataset(Dataset):
    """
    Each item:
      X: (max_frames, D) float32
      Y: (U,) long (targets)
      input_len: int (valid frames before padding/trunc)
      target_len: int (U)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        landmarks_dir: str,
        max_frames: int = MAX_FRAMES_DEFAULT,
        use_delta_features: bool = False,
        normalize_landmarks: bool = False,
        landmark_scale_mode: str = "median_radius",
    ):
        self.df = df.reset_index(drop=True)
        self.landmarks_dir = landmarks_dir
        self.max_frames = max_frames
        self.use_delta_features = use_delta_features
        self.normalize_landmarks = normalize_landmarks
        self.landmark_scale_mode = landmark_scale_mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_id = int(row["file_id"])
        sequence_id = int(row["sequence_id"])

        if "landmarks_subdir" in row and pd.notna(row["landmarks_subdir"]):
            parquet_path = os.path.join(self.landmarks_dir, str(row["landmarks_subdir"]), f"{file_id}.parquet")
        else:
            parquet_path = os.path.join(self.landmarks_dir, f"{file_id}.parquet")
        if not os.path.exists(parquet_path):
            # If parquet missing, mark sample invalid
            return None

        X_raw = read_right_hand_sequence(parquet_path, sequence_id)  # (T, D)
        input_len = count_valid_frames(X_raw)

        if self.normalize_landmarks:
            X_raw = _center_and_scale_frames(X_raw, landmark_scale_mode=self.landmark_scale_mode)

        X_raw = np.nan_to_num(X_raw, nan=0.0)
        if self.use_delta_features:
            X_raw = _append_delta_features(X_raw)

        X = normalize_frames(X_raw, self.max_frames)

        Y = torch.tensor(row["encoded"], dtype=torch.long)
        target_len = int(len(Y))

        # CTC requirement: input_len >= target_len (very strict when input_len small)
        if input_len < target_len or target_len == 0:
            return None

        X = torch.tensor(X, dtype=torch.float32)
        return X, Y, int(min(input_len, self.max_frames)), target_len


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    Xs, Ys, in_lens, tar_lens = [], [], [], []
    for X, Y, in_len, tar_len in batch:
        Xs.append(X)
        Ys.append(Y)
        in_lens.append(in_len)
        tar_lens.append(tar_len)

    Xs = torch.stack(Xs)               # (B, T, D)
    Ys = torch.cat(Ys)                 # (sum_U,)
    in_lens = torch.tensor(in_lens, dtype=torch.long)
    tar_lens = torch.tensor(tar_lens, dtype=torch.long)

    return Xs, Ys, in_lens, tar_lens
