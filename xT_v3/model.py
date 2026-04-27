"""
xT_v3 two-stage model.

Stage 1  — Frozen xT_v2 CNN (position-based threat estimate).
Stage 2  — Small Transformer that reads player tokens from a 360 freeze frame
           and outputs a residual adjustment on top of Stage 1's logit.

Forward pass returns a single logit (pre-sigmoid), same as xT_v2, so the same
BCEWithLogitsLoss training loop applies.
"""
import sys
import os
import importlib.util

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Resolve imports
# ---------------------------------------------------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.join(_HERE, '..', 'xT_v2')
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

from model import XTModel  # noqa: E402 — v2 CNN

# Load v3 config by file path to avoid v2/config shadowing it on sys.path
_spec  = importlib.util.spec_from_file_location("_v3config", os.path.join(_HERE, "config.py"))
_v3cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3cfg)

PLAYER_DIM       = _v3cfg.PLAYER_DIM
MAX_PLAYERS      = _v3cfg.MAX_PLAYERS
BALL_DIM         = _v3cfg.BALL_DIM
BALL_FEAT_INDICES = _v3cfg.BALL_FEAT_INDICES
D_MODEL          = _v3cfg.D_MODEL
N_HEADS          = _v3cfg.N_HEADS
N_LAYERS         = _v3cfg.N_LAYERS
SCALAR_DIM       = _v3cfg.SCALAR_DIM


# ---------------------------------------------------------------------------
# Stage 2: context-adjustment Transformer
# ---------------------------------------------------------------------------

class Stage2Model(nn.Module):
    """
    Reads player tokens from a 360 freeze frame and produces a scalar
    residual adjustment for Stage 1's threat logit.

    Inputs
    ------
    stage1_logit  : (B, 1)              — pre-sigmoid logit from frozen Stage 1
    ball_feats    : (B, BALL_DIM)       — [start_x_norm, start_y_norm, dist_to_goal, angle_to_goal]
    player_tokens : (B, MAX_PLAYERS, PLAYER_DIM)
                    — [x_norm, y_norm, is_teammate, is_keeper, is_actor] per player row
    player_mask   : (B, MAX_PLAYERS) bool
                    — True = padding position (player slot is empty); ignored in attention

    Output
    ------
    adjusted_logit : (B, 1)   — stage1_logit + learned context adjustment
    """

    def __init__(self):
        super().__init__()

        # Project player tokens and ball query into D_MODEL space
        self.player_embed = nn.Linear(PLAYER_DIM, D_MODEL)
        self.ball_embed   = nn.Linear(BALL_DIM,   D_MODEL)

        # Player self-attention: players reason about each other's positions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.player_encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # Cross-attention: ball position queries the player context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D_MODEL,
            num_heads=N_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        # Final MLP: fuses attended player context with Stage 1 logit
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL + 1, 64),  # +1 for stage1_logit
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # Small init so Stage 2 starts as a near-zero correction
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        stage1_logit:  torch.Tensor,   # (B,) or (B, 1)
        ball_feats:    torch.Tensor,   # (B, BALL_DIM)
        player_tokens: torch.Tensor,   # (B, MAX_PLAYERS, PLAYER_DIM)
        player_mask:   torch.Tensor,   # (B, MAX_PLAYERS) bool
    ) -> torch.Tensor:

        # Normalise stage1_logit to (B, 1)
        if stage1_logit.dim() == 1:
            stage1_logit = stage1_logit.unsqueeze(1)

        # --- Embed players ---
        player_embeds = self.player_embed(player_tokens)        # (B, P, D)

        # Self-attention among players (mask = True means "ignore this position")
        player_context = self.player_encoder(
            player_embeds,
            src_key_padding_mask=player_mask,
        )                                                        # (B, P, D)

        # --- Ball queries player context ---
        ball_query = self.ball_embed(ball_feats).unsqueeze(1)   # (B, 1, D)
        attn_out, _ = self.cross_attn(
            query=ball_query,
            key=player_context,
            value=player_context,
            key_padding_mask=player_mask,
        )                                                        # (B, 1, D)
        attn_out = attn_out.squeeze(1)                          # (B, D)

        # --- Fuse with Stage 1 logit ---
        combined   = torch.cat([attn_out, stage1_logit], dim=-1)  # (B, D+1)
        adjustment = self.mlp(combined)                            # (B, 1)

        return (stage1_logit + adjustment).squeeze(1)             # (B,) — matches v2 convention


# ---------------------------------------------------------------------------
# Full xT_v3 model (Stage 1 frozen + Stage 2 trainable)
# ---------------------------------------------------------------------------

class XTModelV3(nn.Module):
    """
    Wraps a frozen xT_v2 CNN (Stage 1) with a trainable Stage 2 Transformer.

    Only Stage 2 parameters are updated during training.
    """

    def __init__(self, stage1: XTModel):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = Stage2Model()

        # Freeze Stage 1 completely
        for param in self.stage1.parameters():
            param.requires_grad = False

    def forward(
        self,
        spatial:       torch.Tensor,   # (B, NUM_CHANNELS, GRID_H, GRID_W)
        scalar:        torch.Tensor,   # (B, SCALAR_DIM)
        player_tokens: torch.Tensor,   # (B, MAX_PLAYERS, PLAYER_DIM)
        player_mask:   torch.Tensor,   # (B, MAX_PLAYERS) bool
    ) -> torch.Tensor:

        with torch.no_grad():
            stage1_logit = self.stage1(spatial, scalar)     # (B,)

        ball_feats = scalar[:, BALL_FEAT_INDICES]           # (B, BALL_DIM)

        return self.stage2(stage1_logit, ball_feats, player_tokens, player_mask)
