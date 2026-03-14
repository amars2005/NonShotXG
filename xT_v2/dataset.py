import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import TRAIN_SPLIT, BATCH_SIZE, NUM_WORKERS


class XTDataset(Dataset):
    """
    PyTorch Dataset wrapping the pre-encoded spatial tensors, scalar
    feature vectors, and binary chain-goal labels.
    """

    def __init__(self, spatial: np.ndarray, scalar: np.ndarray, labels: np.ndarray):
        self.spatial = torch.from_numpy(spatial).float()
        self.scalar  = torch.from_numpy(scalar).float()
        self.labels  = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.spatial[idx], self.scalar[idx], self.labels[idx]


def make_dataloaders(
    spatial:   np.ndarray,
    scalar:    np.ndarray,
    labels:    np.ndarray,
    match_ids: np.ndarray,
) -> tuple[DataLoader, DataLoader]:
    """
    Split the dataset by match ID (to prevent data leakage across the
    train/val boundary) and return DataLoaders for both splits.

    The split is deterministic (seed=42) and respects the TRAIN_SPLIT
    ratio defined in config.py.

    Parameters
    ----------
    spatial   : (N, C, H, W) float32 array
    scalar    : (N, SCALAR_DIM) float32 array
    labels    : (N,) float32 array
    match_ids : (N,) int array  — one entry per event

    Returns
    -------
    train_loader, val_loader
    """
    unique_matches = np.unique(match_ids)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(unique_matches)

    split        = int(len(unique_matches) * TRAIN_SPLIT)
    train_set    = set(unique_matches[:split].tolist())
    val_set      = set(unique_matches[split:].tolist())

    train_mask = np.array([m in train_set for m in match_ids])
    val_mask   = np.array([m in val_set   for m in match_ids])

    print(
        f"Train: {train_mask.sum():,} events ({len(train_set)} matches)  |  "
        f"Val: {val_mask.sum():,} events ({len(val_set)} matches)"
    )

    train_ds = XTDataset(spatial[train_mask], scalar[train_mask], labels[train_mask])
    val_ds   = XTDataset(spatial[val_mask],   scalar[val_mask],   labels[val_mask])

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    return train_loader, val_loader
