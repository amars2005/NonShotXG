import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import GRID_W, GRID_H, PITCH_W, PITCH_H, NUM_CHANNELS, HEATMAP_PATH
from encoder import encode_event, build_inference_scalar


# ---------------------------------------------------------------------------
# Scenario definitions
# All coordinates are in StatsBomb space: x ∈ [0,120], y ∈ [0,80]
# Attacking direction is left → right (goal at x=120, y=40)
# 'teammate' = same team as ball carrier (attackers)
# 'keeper'   = opponent goalkeeper
# ---------------------------------------------------------------------------

def _p(x, y, teammate=False, keeper=False):
    return {'location': [x, y], 'teammate': teammate, 'actor': False, 'keeper': keeper}

SCENARIOS = {
    'No Players (Baseline)': None,

    'Compact Low Block': {
        'players': [
            _p(119, 40, keeper=True),
            # Back four
            _p(102, 20), _p(102, 33), _p(102, 47), _p(102, 60),
            # Mid four
            _p(88, 22),  _p(88, 35),  _p(88, 45),  _p(88, 58),
        ],
        'visible_area': [],
    },

    'High Press': {
        'players': [
            _p(119, 40, keeper=True),
            # Mid four pressing high into opponent half
            _p(62, 22), _p(62, 35), _p(62, 45), _p(62, 58),
            # Back four pushed up to halfway
            _p(72, 20), _p(72, 33), _p(72, 47), _p(72, 60),
        ],
        'visible_area': [],
    },

    '2v1 Overload in Box': {
        'players': [
            _p(119, 40, keeper=True),
            # 1 defender in box
            _p(107, 40),
            # 2 attacking teammates making runs
            _p(108, 28, teammate=True),
            _p(108, 52, teammate=True),
            # Covering defenders wide
            _p(103, 18), _p(103, 62),
        ],
        'visible_area': [],
    },

    'Counter-Attack 3v2': {
        'players': [
            _p(119, 40, keeper=True),
            # 2 covering defenders caught between ball and runners
            _p(105, 30), _p(105, 50),
            # Wide runners already in behind the defence
            _p(110, 22, teammate=True),
            _p(110, 58, teammate=True),
            # Central runner linking play (ball carrier area)
            _p(92, 40, teammate=True),
        ],
        'visible_area': [],
    },

    'Goalkeeper Off Line': {
        'players': [
            # Keeper rushed out to edge of box (~18 yards from goal)
            _p(106, 40, keeper=True),
            # Back four sitting deeper than keeper, covering behind
            _p(100, 22), _p(100, 35), _p(100, 45), _p(100, 58),
        ],
        'visible_area': [],
    },
}


def _sweep_pitch(model, device, frame_data, action_type):
    """Run ball-position sweep across full pitch with fixed player layout."""
    x_centres = np.linspace(PITCH_W / (2 * GRID_W), PITCH_W - PITCH_W / (2 * GRID_W), GRID_W)
    y_centres = np.linspace(PITCH_H / (2 * GRID_H), PITCH_H - PITCH_H / (2 * GRID_H), GRID_H)
    xx, yy = np.meshgrid(x_centres, y_centres)

    from config import SCALAR_DIM
    n_points = GRID_H * GRID_W
    spatial_batch = np.zeros((n_points, NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
    scalar_batch  = np.zeros((n_points, SCALAR_DIM), dtype=np.float32)

    idx = 0
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            sx, sy = float(xx[gy, gx]), float(yy[gy, gx])
            event_row = {'start_x': sx, 'start_y': sy, 'end_x': np.nan, 'end_y': np.nan}
            spatial_batch[idx] = encode_event(event_row, frame_data)
            scalar_batch[idx]  = build_inference_scalar(sx, sy, action_type)
            idx += 1

    all_probs = []
    with torch.no_grad():
        for start in range(0, n_points, 256):
            end    = min(start + 256, n_points)
            sp     = torch.from_numpy(spatial_batch[start:end]).to(device)
            sc     = torch.from_numpy(scalar_batch[start:end]).to(device)
            logits = model(sp, sc)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

    return xx, yy, np.array(all_probs, dtype=np.float32).reshape(GRID_H, GRID_W)


def generate_heatmap(
    model:       nn.Module,
    device:      torch.device,
    action_type: str = 'Pass',
) -> None:
    """
    Sweeps ball position across every cell of the pitch grid, runs each
    through the model (with empty player channels and full visibility),
    and saves a contour heatmap of the predicted xT values.

    Parameters
    ----------
    model       : trained XTModel (weights already loaded)
    device      : torch.device
    action_type : the StatsBomb action type to simulate ('Pass' by default)
    """
    print(f"Generating xT heatmap for action type: '{action_type}'...")
    model.eval()
    xx, yy, z = _sweep_pitch(model, device, frame_data=None, action_type=action_type)
    _plot_heatmap(xx, yy, z, action_type)


def generate_all_scenarios(
    model:       nn.Module,
    device:      torch.device,
    action_type: str = 'Pass',
) -> None:
    """
    Generate a multi-panel heatmap comparing all defined player scenarios.
    Saved alongside the standard heatmap as xt_scenarios_v2.png.
    """
    print(f"Generating scenario heatmaps for action type: '{action_type}'...")
    model.eval()

    n = len(SCENARIOS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5))
    axes = axes.flatten()

    global_max = 0.0
    results = {}
    for name, frame_data in SCENARIOS.items():
        xx, yy, z = _sweep_pitch(model, device, frame_data, action_type)
        results[name] = (xx, yy, z)
        global_max = max(global_max, z.max())

    for ax, (name, (xx, yy, z)) in zip(axes, results.items()):
        cf = ax.contourf(xx, yy, z, levels=30, cmap='magma', vmin=0, vmax=global_max)
        fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
        _draw_pitch(ax)

        # Overlay player markers for non-baseline scenarios
        frame_data = SCENARIOS[name]
        if frame_data and 'players' in frame_data:
            for p in frame_data['players']:
                px, py = p['location']
                if p.get('keeper'):
                    ax.scatter(px, py, c='cyan',   s=80, zorder=6, marker='s', label='GK')
                elif p.get('teammate'):
                    ax.scatter(px, py, c='lime',   s=60, zorder=6, marker='^', label='Attacker')
                else:
                    ax.scatter(px, py, c='red',    s=60, zorder=6, marker='o', label='Defender')

        ax.set_title(name, fontsize=11, pad=6)
        ax.set_xlim(0, PITCH_W)
        ax.set_ylim(PITCH_H, 0)
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f'Non-Shot xT — Scenario Comparison  (action={action_type})',
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    slug = action_type.replace(' ', '_').replace('*', '').lower()
    out_path = os.path.join(os.path.dirname(HEATMAP_PATH), f'xt_scenarios_{slug}_v2.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Scenario heatmaps saved to {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_heatmap(xx: np.ndarray, yy: np.ndarray, z: np.ndarray, action_type: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    # Heatmap layer
    cf = ax.contourf(xx, yy, z, levels=30, cmap='magma', vmin=0, vmax=z.max())
    cbar = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('P(chain → goal)  —  xT', fontsize=11)

    # Pitch markings on top
    _draw_pitch(ax)

    ax.set_title(
        f'Non-Shot Expected Threat Map  (CNN, action={action_type})',
        fontsize=14, pad=12,
    )
    ax.set_xlim(0, PITCH_W)
    ax.set_ylim(PITCH_H, 0)   # Invert Y: (0,0) top-left is standard football view
    ax.set_aspect('equal')
    ax.axis('off')

    slug = action_type.replace(' ', '_').replace('*', '').lower()
    out_path = HEATMAP_PATH.replace('_v2.png', f'_{slug}_v2.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap saved to {out_path}")


def _draw_pitch(ax: plt.Axes) -> None:
    """
    Draw standard football pitch markings on StatsBomb coordinates (120 × 80).
    All lines are white so they show clearly over the dark magma colormap.
    """
    lc = 'white'
    lw = 1.5

    # Outline & halfway line
    ax.plot([0,   0],   [0,  80], color=lc, linewidth=lw)
    ax.plot([120, 120], [0,  80], color=lc, linewidth=lw)
    ax.plot([0,  120],  [0,   0], color=lc, linewidth=lw)
    ax.plot([0,  120],  [80, 80], color=lc, linewidth=lw)
    ax.plot([60,  60],  [0,  80], color=lc, linewidth=lw)

    # Centre circle & spot
    circle = mpatches.Circle((60, 40), 10, color=lc, fill=False, linewidth=lw)
    ax.add_patch(circle)
    ax.scatter(60, 40, color=lc, s=15, zorder=5)

    # Penalty areas (18-yard boxes)
    ax.plot([18,  18],  [18, 62], color=lc, linewidth=lw)
    ax.plot([0,   18],  [18, 18], color=lc, linewidth=lw)
    ax.plot([0,   18],  [62, 62], color=lc, linewidth=lw)
    ax.plot([102, 102], [18, 62], color=lc, linewidth=lw)
    ax.plot([102, 120], [18, 18], color=lc, linewidth=lw)
    ax.plot([102, 120], [62, 62], color=lc, linewidth=lw)

    # 6-yard boxes
    ax.plot([6,   6],   [30, 50], color=lc, linewidth=lw)
    ax.plot([0,   6],   [30, 30], color=lc, linewidth=lw)
    ax.plot([0,   6],   [50, 50], color=lc, linewidth=lw)
    ax.plot([114, 114], [30, 50], color=lc, linewidth=lw)
    ax.plot([114, 120], [30, 30], color=lc, linewidth=lw)
    ax.plot([114, 120], [50, 50], color=lc, linewidth=lw)

    # Penalty spots
    ax.scatter(12,  40, color=lc, s=15, zorder=5)
    ax.scatter(108, 40, color=lc, s=15, zorder=5)
