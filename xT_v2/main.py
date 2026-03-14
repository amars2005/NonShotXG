"""
xT_v2 — Non-Shot Expected Threat model with StatsBomb 360 data and CNN.

Usage
-----
  python main.py --build              # Encode all 360-compatible matches to disk
  python main.py --build --limit 20   # Quick test: encode only 20 matches
  python main.py --build --force      # Re-encode even if .npz already exists
  python main.py --train              # Train the CNN on the built dataset
  python main.py --visualize          # Generate xT heatmap from best checkpoint
  python main.py --all                # Build → Train → Visualize in sequence
"""

import argparse
import torch

from config import BEST_MODEL_PATH


def _get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    return device


def run_build(limit: int | None, force: bool) -> None:
    from builder import DatasetBuilder
    builder = DatasetBuilder()
    builder.build(limit=limit, force_rebuild=force)


def run_train(device: torch.device) -> None:
    from builder import DatasetBuilder
    from dataset import make_dataloaders
    from model import XTModel
    from train import train

    print("\n--- Loading dataset ---")
    builder = DatasetBuilder()
    spatial, scalar, labels, match_ids = builder.load_all()
    print(f"Total events: {len(labels):,}  |  Goal rate: {labels.mean():.4f}")

    train_loader, val_loader = make_dataloaders(spatial, scalar, labels, match_ids)

    print("\n--- Building model ---")
    model = XTModel().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    print("\n--- Training ---")
    train(model, train_loader, val_loader, device)


def run_visualize(device: torch.device) -> None:
    import os
    from model import XTModel
    from visualize import generate_heatmap

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {BEST_MODEL_PATH}.\n"
            "Run  python main.py --train  first."
        )

    model = XTModel().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    print("\n--- Generating heatmap ---")
    generate_heatmap(model, device, action_type='Pass')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xT_v2 pipeline")
    parser.add_argument("--build",    action="store_true", help="Encode dataset from StatsBomb API")
    parser.add_argument("--train",    action="store_true", help="Train the CNN model")
    parser.add_argument("--visualize",action="store_true", help="Generate xT heatmap from checkpoint")
    parser.add_argument("--all",      action="store_true", help="Run full pipeline: build → train → visualize")
    parser.add_argument("--limit",    type=int, default=None, help="Cap number of matches for --build")
    parser.add_argument("--force",    action="store_true",    help="Force re-encode in --build")
    args = parser.parse_args()

    # Default to --all if no flags given
    if not any([args.build, args.train, args.visualize, args.all]):
        args.all = True

    device = _get_device()

    if args.build or args.all:
        print("\n=== STEP 1: BUILD DATASET ===")
        run_build(limit=args.limit, force=args.force)

    if args.train or args.all:
        print("\n=== STEP 2: TRAIN MODEL ===")
        run_train(device)

    if args.visualize or args.all:
        print("\n=== STEP 3: VISUALIZE ===")
        run_visualize(device)
