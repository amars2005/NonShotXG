import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import TRAIN_SPLIT, VAL_SPLIT, BATCH_SIZE, NUM_WORKERS


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
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split the dataset by match ID (to prevent data leakage) and return
    DataLoaders for train, val, and held-out test splits.

    Ratios are controlled by TRAIN_SPLIT and VAL_SPLIT in config.py
    (default 70 / 15 / 15).  The split is deterministic (seed=42).
    """
    unique_matches = np.unique(match_ids)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(unique_matches)

    train_end = int(len(unique_matches) * TRAIN_SPLIT)
    val_end   = train_end + int(len(unique_matches) * VAL_SPLIT)

    train_set = set(unique_matches[:train_end].tolist())
    val_set   = set(unique_matches[train_end:val_end].tolist())
    test_set  = set(unique_matches[val_end:].tolist())

    train_mask = np.array([m in train_set for m in match_ids])
    val_mask   = np.array([m in val_set   for m in match_ids])
    test_mask  = np.array([m in test_set  for m in match_ids])

    print(
        f"Train: {train_mask.sum():,} events ({len(train_set)} matches)  |  "
        f"Val: {val_mask.sum():,} events ({len(val_set)} matches)  |  "
        f"Test: {test_mask.sum():,} events ({len(test_set)} matches)"
    )

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    train_loader = DataLoader(
        XTDataset(spatial[train_mask], scalar[train_mask], labels[train_mask]),
        shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        XTDataset(spatial[val_mask], scalar[val_mask], labels[val_mask]),
        shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        XTDataset(spatial[test_mask], scalar[test_mask], labels[test_mask]),
        shuffle=False, **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def make_test_loader(
    spatial:          np.ndarray,
    scalar:           np.ndarray,
    labels:           np.ndarray,
    match_ids:        np.ndarray,
    target_match_ids: set,
) -> DataLoader:
    """
    Build a test DataLoader restricted to a specific set of match IDs.

    Used by main.py to re-scope v2's test evaluation to the same freeze-frame
    matches that xT_v3 was evaluated on, so the two models are directly
    comparable on the same event population.
    """
    mask = np.array([m in target_match_ids for m in match_ids])
    print(
        f"Test (freeze-frame scope): {mask.sum():,} events "
        f"({len(target_match_ids)} matches)"
    )
    return DataLoader(
        XTDataset(spatial[mask], scalar[mask], labels[mask]),
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
