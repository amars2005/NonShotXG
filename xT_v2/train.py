import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from config import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BEST_MODEL_PATH, CHECKPOINT_DIR, METRICS_PATH

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from evaluate import compute_metrics, print_metrics, save_metrics

LOG_PATH = os.path.join(os.path.dirname(CHECKPOINT_DIR), 'train.log')


def _log(msg: str) -> None:
    """Print to stdout, flushed immediately. tee handles the log file."""
    print(msg, flush=True)


def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    test_loader:  DataLoader | None = None,
) -> nn.Module:
    """
    Full training loop with:
      - BCEWithLogitsLoss
      - Adam optimiser with L2 weight decay
      - ReduceLROnPlateau scheduler (monitors val AUC)
      - Checkpoint saving on best validation AUC

    Parameters
    ----------
    model        : XTModel instance (already moved to device)
    train_loader : DataLoader for training set
    val_loader   : DataLoader for validation set
    device       : torch.device

    Returns
    -------
    model with weights from the best checkpoint loaded.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Compute positive class weight from training labels to handle severe class imbalance.
    # Goal rate is ~0.5%, so without this the model collapses to predicting all zeros.
    train_labels = train_loader.dataset.labels
    pos_rate = train_labels.mean().item()
    # Soft labels already encode threat magnitude, so a modest pos_weight is enough.
    # The raw ratio (~315) over-amplifies goal-chain gradients and collapses Spearman.
    pos_weight = torch.tensor([min((1.0 - pos_rate) / (pos_rate + 1e-8), 10.0)]).to(device)
    _log(f"Class balance — goal rate: {pos_rate:.4f}  |  pos_weight: {pos_weight.item():.1f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    best_val_spearman = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):

        # ---- Training ------------------------------------------------
        model.train()
        train_loss = 0.0

        for spatial, scalar, labels in train_loader:
            spatial = spatial.to(device, non_blocking=True)
            scalar  = scalar.to(device,  non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)

            optimizer.zero_grad()
            logits = model(spatial, scalar)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)

        train_loss /= len(train_loader.dataset)

        # ---- Validation ----------------------------------------------
        val_spearman, val_mse, val_loss = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_spearman)

        _log(
            f"Epoch {epoch:>3}/{NUM_EPOCHS}  |  "
            f"Train Loss: {train_loss:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  |  "
            f"Val Spearman: {val_spearman:.4f}  |  "
            f"Val MSE: {val_mse:.4f}"
        )

        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            _log(f"  -> Saved best model  (Spearman ρ {val_spearman:.4f})")

    _log(f"\nTraining complete.  Best Val Spearman ρ: {best_val_spearman:.4f}")

    # Reload best weights before returning
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    if test_loader is not None:
        _log("\nRunning final evaluation on held-out test set...")
        probs, labels = _get_predictions(model, test_loader, device)
        metrics = compute_metrics(labels, probs)
        print_metrics(metrics, "Test Set (v2 CNN)")
        save_metrics(metrics, METRICS_PATH)

    return model


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, float]:
    """
    Run inference over a DataLoader and return (Spearman ρ, MSE, mean loss).

    Spearman ρ measures rank correlation between predicted threat and soft
    discounted labels — appropriate since xT is a ranking problem.
    MSE measures calibration against the soft label targets directly.
    """
    model.eval()
    all_probs:  list[float] = []
    all_labels: list[float] = []
    total_loss = 0.0

    with torch.no_grad():
        for spatial, scalar, labels in loader:
            spatial = spatial.to(device, non_blocking=True)
            scalar  = scalar.to(device,  non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)

            logits = model(spatial, scalar)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_probs  = np.array(all_probs,  dtype=np.float64)
    all_labels = np.array(all_labels, dtype=np.float64)

    spearman  = spearmanr(all_labels, all_probs).statistic
    mse       = np.mean((all_labels - all_probs) ** 2)
    mean_loss = total_loss / len(loader.dataset)

    return spearman, mse, mean_loss


def _get_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) arrays for the full loader — used for final test evaluation."""
    model.eval()
    all_probs:  list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for spatial, scalar, labels in loader:
            spatial = spatial.to(device, non_blocking=True)
            scalar  = scalar.to(device,  non_blocking=True)
            logits  = model(spatial, scalar)
            probs   = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    return (
        np.array(all_probs,  dtype=np.float64),
        np.array(all_labels, dtype=np.float64),
    )
