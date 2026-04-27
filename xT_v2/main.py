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
import json
import os

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
    from dataset import make_dataloaders, make_test_loader
    from model import XTModel
    from train import train

    print("\n--- Loading dataset ---")
    builder = DatasetBuilder()
    spatial, scalar, labels, match_ids = builder.load_all()
    print(f"Total events: {len(labels):,}  |  Goal rate: {labels.mean():.4f}")

    train_loader, val_loader, test_loader = make_dataloaders(spatial, scalar, labels, match_ids)

    # If xT_v3 has already been built, use its freeze-frame test match IDs so
    # v2 and v3 are evaluated on exactly the same event population.
    _v3_ids_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "xT_v3", "checkpoints", "test_match_ids.json",
    )
    if os.path.exists(_v3_ids_path):
        with open(_v3_ids_path) as f:
            v3_test_ids = set(json.load(f))
        print(f"\nFound v3 test match IDs → rescoping v2 test set to freeze-frame matches.")
        test_loader = make_test_loader(spatial, scalar, labels, match_ids, v3_test_ids)
    else:
        print(
            "\nNo v3 test_match_ids.json found — using v2's own test split.\n"
            "Re-run  python main.py --train  after building xT_v3 to align test scopes."
        )

    print("\n--- Building model ---")
    model = XTModel().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    print("\n--- Training ---")
    train(model, train_loader, val_loader, device, test_loader=test_loader)


def run_visualize(device: torch.device) -> None:
    import os
    from model import XTModel
    from visualize import generate_heatmap, generate_all_scenarios

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {BEST_MODEL_PATH}.\n"
            "Run  python main.py --train  first."
        )

    model = XTModel().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    for action_type in ('Pass', 'Carry', 'Dribble'):
        print(f"\n--- Generating heatmap: {action_type} ---")
        generate_heatmap(model, device, action_type=action_type)
        generate_all_scenarios(model, device, action_type=action_type)


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
