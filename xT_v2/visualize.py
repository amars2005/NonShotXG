import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import GRID_W, GRID_H, PITCH_W, PITCH_H, NUM_CHANNELS, HEATMAP_PATH
from encoder import encode_event, build_inference_scalar


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

    # Build a grid of pitch-centre coordinates for each cell
    x_centres = np.linspace(PITCH_W / (2 * GRID_W), PITCH_W - PITCH_W / (2 * GRID_W), GRID_W)
    y_centres = np.linspace(PITCH_H / (2 * GRID_H), PITCH_H - PITCH_H / (2 * GRID_H), GRID_H)
    xx, yy = np.meshgrid(x_centres, y_centres)  # both (GRID_H, GRID_W)

    n_points = GRID_H * GRID_W
    spatial_batch = np.zeros((n_points, NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
    scalar_batch  = np.zeros((n_points, ),             dtype=np.float32)  # placeholder shape
    from config import SCALAR_DIM
    scalar_batch  = np.zeros((n_points, SCALAR_DIM),   dtype=np.float32)

    idx = 0
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            sx = float(xx[gy, gx])
            sy = float(yy[gy, gx])

            event_row = {'start_x': sx, 'start_y': sy, 'end_x': np.nan, 'end_y': np.nan}
            # No frame_data -> empty player channels, full visible area assumed
            spatial_batch[idx] = encode_event(event_row, frame_data=None)
            scalar_batch[idx]  = build_inference_scalar(sx, sy, action_type)
            idx += 1

    # Run model in batches to avoid OOM on large grids
    batch_size = 256
    all_probs  = []

    with torch.no_grad():
        for start in range(0, n_points, batch_size):
            end     = min(start + batch_size, n_points)
            sp      = torch.from_numpy(spatial_batch[start:end]).to(device)
            sc      = torch.from_numpy(scalar_batch[start:end]).to(device)
            logits  = model(sp, sc)
            probs   = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

    z = np.array(all_probs, dtype=np.float32).reshape(GRID_H, GRID_W)

    # Map back to full pitch coordinates for plotting
    plot_xx = xx
    plot_yy = yy

    _plot_heatmap(plot_xx, plot_yy, z, action_type)


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

    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap saved to {HEATMAP_PATH}")


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
