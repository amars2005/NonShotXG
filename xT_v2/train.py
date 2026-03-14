import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss

from config import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BEST_MODEL_PATH, CHECKPOINT_DIR


def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
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
    train_labels = torch.cat([labels for _, _, labels in train_loader])
    pos_rate = train_labels.mean().item()
    pos_weight = torch.tensor([(1.0 - pos_rate) / (pos_rate + 1e-8)]).to(device)
    print(f"Class balance — goal rate: {pos_rate:.4f}  |  pos_weight: {pos_weight.item():.1f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    best_val_auc = 0.0

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
        val_auc, val_brier, val_loss = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)

        print(
            f"Epoch {epoch:>3}/{NUM_EPOCHS}  |  "
            f"Train Loss: {train_loss:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  |  "
            f"Val AUC: {val_auc:.4f}  |  "
            f"Val Brier: {val_brier:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Saved best model  (AUC {val_auc:.4f})")

    print(f"\nTraining complete.  Best Val AUC: {best_val_auc:.4f}")

    # Reload best weights before returning
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
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
    Run inference over a DataLoader and return (AUC, Brier score, mean loss).
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

    auc   = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    brier = brier_score_loss(all_labels, all_probs)
    mean_loss = total_loss / len(loader.dataset)

    return auc, brier, mean_loss
