import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

_spec  = importlib.util.spec_from_file_location("_v3config", os.path.join(_HERE, "config.py"))
_v3cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3cfg)

TRAIN_SPLIT       = _v3cfg.TRAIN_SPLIT
VAL_SPLIT         = _v3cfg.VAL_SPLIT
BATCH_SIZE_V3     = _v3cfg.BATCH_SIZE_V3
NUM_WORKERS       = _v3cfg.NUM_WORKERS
MAX_PLAYERS       = _v3cfg.MAX_PLAYERS
PLAYER_DIM        = _v3cfg.PLAYER_DIM
CHECKPOINT_DIR_V3 = _v3cfg.CHECKPOINT_DIR_V3
TEST_MATCH_IDS_PATH = _v3cfg.TEST_MATCH_IDS_PATH


class XTDatasetV3(Dataset):
    """
    PyTorch Dataset for the two-stage xT_v3 model.

    Each item is a tuple of:
        spatial        : (NUM_CHANNELS, GRID_H, GRID_W)    float32
        scalar         : (SCALAR_DIM,)                     float32
        player_tokens  : (MAX_PLAYERS, PLAYER_DIM)          float32
        player_mask    : (MAX_PLAYERS,)                     bool
        label          : ()                                float32
    """

    def __init__(
        self,
        spatial:       np.ndarray,   # (N, C, H, W)
        scalar:        np.ndarray,   # (N, SCALAR_DIM)
        player_tokens: np.ndarray,   # (N, MAX_PLAYERS, PLAYER_DIM)
        player_mask:   np.ndarray,   # (N, MAX_PLAYERS) bool
        labels:        np.ndarray,   # (N,)
    ):
        self.spatial       = torch.from_numpy(spatial).float()
        self.scalar        = torch.from_numpy(scalar).float()
        self.player_tokens = torch.from_numpy(player_tokens).float()
        self.player_mask   = torch.from_numpy(player_mask.astype(bool))
        self.labels        = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.spatial[idx],
            self.scalar[idx],
            self.player_tokens[idx],
            self.player_mask[idx],
            self.labels[idx],
        )


def make_dataloaders_v3(
    spatial:       np.ndarray,
    scalar:        np.ndarray,
    player_tokens: np.ndarray,
    player_mask:   np.ndarray,
    labels:        np.ndarray,
    match_ids:     np.ndarray,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split by match ID (no leakage) and return train / val / test DataLoaders.

    Uses the same seed=42 shuffle and 70/15/15 split as v1 and v2 for
    consistent cross-version comparison.  Saves test match IDs to
    TEST_MATCH_IDS_PATH so xT_v2 can re-scope its own test evaluation to
    the same freeze-frame event population.
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

    # Persist test match IDs so v2 can align its test scope to these
    # freeze-frame matches when re-run after v3 is built.
    os.makedirs(CHECKPOINT_DIR_V3, exist_ok=True)
    with open(TEST_MATCH_IDS_PATH, "w") as f:
        json.dump(sorted(test_set), f)
    print(f"Test match IDs saved → {TEST_MATCH_IDS_PATH}")

    loader_kwargs = dict(batch_size=BATCH_SIZE_V3, num_workers=NUM_WORKERS, pin_memory=True)

    def _make(mask):
        return XTDatasetV3(
            spatial[mask], scalar[mask],
            player_tokens[mask], player_mask[mask], labels[mask],
        )

    train_loader = DataLoader(_make(train_mask), shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(_make(val_mask),   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(_make(test_mask),  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
