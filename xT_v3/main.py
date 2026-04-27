"""
xT_v3 — Two-stage Non-Shot Expected Threat model.

Stage 1: Frozen xT_v2 CNN  (position-based threat estimate)
Stage 2: Transformer cross-attention over 360 freeze-frame player tokens

Usage
-----
  python main.py --build              # Encode 360-compatible matches (freeze-frame events only)
  python main.py --build --limit 20   # Quick test: encode only 20 matches
  python main.py --build --force      # Re-encode even if .npz already exists
  python main.py --train              # Train Stage 2 (Stage 1 weights must exist)
  python main.py --visualize          # Generate heatmaps from best_model.pt
  python main.py --all                # Build → Train in sequence
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import BEST_MODEL_PATH_V3, V2_BEST_MODEL_PATH


def _get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    return device


def run_build(limit: int | None, force: bool) -> None:
    from builder import DatasetBuilderV3
    builder = DatasetBuilderV3()
    builder.build(limit=limit, force_rebuild=force)


def run_train(device: torch.device) -> None:
    import importlib.util

    _V2_DIR = os.path.join(_HERE, '..', 'xT_v2')
    if _V2_DIR not in sys.path:
        sys.path.insert(0, _V2_DIR)

    # Load v2 XTModel by file path
    _spec_v2m = importlib.util.spec_from_file_location("_v2model", os.path.join(_V2_DIR, "model.py"))
    _v2m      = importlib.util.module_from_spec(_spec_v2m)
    _spec_v2m.loader.exec_module(_v2m)
    XTModel = _v2m.XTModel

    # Load v3 model by file path
    _spec_v3m = importlib.util.spec_from_file_location("_v3model", os.path.join(_HERE, "model.py"))
    _v3m      = importlib.util.module_from_spec(_spec_v3m)
    _spec_v3m.loader.exec_module(_v3m)
    XTModelV3 = _v3m.XTModelV3

    # Load v3 builder/dataset/train by file path to avoid hitting v2 versions
    _spec_b = importlib.util.spec_from_file_location("_v3builder", os.path.join(_HERE, "builder.py"))
    _v3b    = importlib.util.module_from_spec(_spec_b); _spec_b.loader.exec_module(_v3b)
    DatasetBuilderV3 = _v3b.DatasetBuilderV3

    _spec_d = importlib.util.spec_from_file_location("_v3dataset", os.path.join(_HERE, "dataset.py"))
    _v3d    = importlib.util.module_from_spec(_spec_d); _spec_d.loader.exec_module(_v3d)
    make_dataloaders_v3 = _v3d.make_dataloaders_v3

    _spec_t = importlib.util.spec_from_file_location("_v3train", os.path.join(_HERE, "train.py"))
    _v3t    = importlib.util.module_from_spec(_spec_t); _spec_t.loader.exec_module(_v3t)
    train_v3 = _v3t.train_v3

    # --- Load v2 Stage 1 ---
    if not os.path.exists(V2_BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at {V2_BEST_MODEL_PATH}.\n"
            "Train xT_v2 first:  cd ../xT_v2 && python main.py --train"
        )

    stage1 = XTModel().to(device)
    stage1.load_state_dict(torch.load(V2_BEST_MODEL_PATH, map_location=device))
    stage1.eval()
    print(f"Stage 1 loaded from {V2_BEST_MODEL_PATH}")

    # --- Build v3 model ---
    model = XTModelV3(stage1).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 2 trainable parameters: {trainable:,}")

    # --- Load dataset ---
    print("\n--- Loading v3 dataset ---")
    builder = DatasetBuilderV3()
    spatial, scalar, player_tokens, player_mask, labels, match_ids = builder.load_all()
    print(f"Total events (with freeze frames): {len(labels):,}  |  Goal rate: {labels.mean():.4f}")

    train_loader, val_loader, test_loader = make_dataloaders_v3(
        spatial, scalar, player_tokens, player_mask, labels, match_ids,
    )

    # --- Train ---
    print("\n--- Training Stage 2 ---")
    train_v3(model, train_loader, val_loader, device, test_loader=test_loader)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xT_v3 pipeline")
    parser.add_argument("--build",     action="store_true", help="Encode dataset (freeze-frame events)")
    parser.add_argument("--train",     action="store_true", help="Train Stage 2 Transformer")
    parser.add_argument("--visualize", action="store_true", help="Generate heatmaps from best checkpoint")
    parser.add_argument("--all",       action="store_true", help="Run full pipeline: build → train")
    parser.add_argument("--limit",     type=int, default=None, help="Cap number of matches for --build")
    parser.add_argument("--force",     action="store_true",    help="Force re-encode in --build")
    args = parser.parse_args()

    if not any([args.build, args.train, args.visualize, args.all]):
        args.all = True

    device = _get_device()

    if args.build or args.all:
        print("\n=== STEP 1: BUILD DATASET (v3) ===")
        run_build(limit=args.limit, force=args.force)

    if args.train or args.all:
        print("\n=== STEP 2: TRAIN STAGE 2 ===")
        run_train(device)

    if args.visualize:
        print("\n=== VISUALIZE (v3) ===")
        import importlib.util as _ilu
        _sv2 = _ilu.spec_from_file_location("_v2model", os.path.join(os.path.dirname(__file__), '..', 'xT_v2', 'model.py'))
        _mv2 = _ilu.module_from_spec(_sv2); _sv2.loader.exec_module(_mv2)
        _sv3 = _ilu.spec_from_file_location("_v3model", os.path.join(os.path.dirname(__file__), 'model.py'))
        _mv3 = _ilu.module_from_spec(_sv3); _sv3.loader.exec_module(_mv3)

        stage1 = _mv2.XTModel().to(device)
        stage1.load_state_dict(torch.load(V2_BEST_MODEL_PATH, map_location=device))
        model  = _mv3.XTModelV3(stage1).to(device)
        model.load_state_dict(torch.load(BEST_MODEL_PATH_V3, map_location=device))
        model.eval()

        _sv = _ilu.spec_from_file_location("_v3viz", os.path.join(os.path.dirname(__file__), 'visualize.py'))
        _vz = _ilu.module_from_spec(_sv); _sv.loader.exec_module(_vz)
        _vz.generate_heatmap(model, device)
        _vz.generate_all_scenarios(model, device)
