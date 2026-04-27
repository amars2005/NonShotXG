"""
xT_v3 dataset builder.

Extends the xT_v2 pipeline by:
  1. Using the same chain-goal labelling logic.
  2. Filtering to ONLY events that have a 360 freeze frame.
  3. Extracting a padded player token array per event.

Each match is saved as a compressed .npz with keys:
  spatial        (N, NUM_CHANNELS, GRID_H, GRID_W)  float32
  scalar         (N, SCALAR_DIM)                    float32
  player_tokens  (N, MAX_PLAYERS, PLAYER_DIM)        float32
  player_mask    (N, MAX_PLAYERS)                    bool
  label          (N,)                               float32
  match_id       (1,)                               int32
"""
import os
import sys
import importlib.util

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Resolve v2 imports via file path to avoid config name collision
# ---------------------------------------------------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.join(_HERE, '..', 'xT_v2')
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

_spec_v2  = importlib.util.spec_from_file_location("_v2config", os.path.join(_V2_DIR, "config.py"))
_v2cfg    = importlib.util.module_from_spec(_spec_v2)
_spec_v2.loader.exec_module(_v2cfg)

SCALAR_COLS   = _v2cfg.SCALAR_COLS
NUM_CHANNELS  = _v2cfg.NUM_CHANNELS
GRID_H        = _v2cfg.GRID_H
GRID_W        = _v2cfg.GRID_W
CHAIN_GAMMA   = _v2cfg.CHAIN_GAMMA
BUILD_WORKERS = _v2cfg.BUILD_WORKERS

from loader  import StatsBomb360Loader   # noqa: E402
from parser  import MatchParser360       # noqa: E402
from encoder import encode_event         # noqa: E402

# Load v3 config by file path
_spec_v3 = importlib.util.spec_from_file_location("_v3config", os.path.join(_HERE, "config.py"))
_v3cfg   = importlib.util.module_from_spec(_spec_v3)
_spec_v3.loader.exec_module(_v3cfg)

DATA_DIR_V3 = _v3cfg.DATA_DIR_V3
MAX_PLAYERS = _v3cfg.MAX_PLAYERS
PLAYER_DIM  = _v3cfg.PLAYER_DIM
PITCH_W     = _v3cfg.PITCH_W
PITCH_H     = _v3cfg.PITCH_H


def _extract_player_tokens(frame_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a freeze-frame dict into a padded player token array.

    Returns
    -------
    tokens : (MAX_PLAYERS, PLAYER_DIM) float32
        Each row: [x_norm, y_norm, is_teammate, is_keeper, is_actor]
    mask   : (MAX_PLAYERS,) bool
        True = padding slot (should be ignored in attention).
    """
    tokens = np.zeros((MAX_PLAYERS, PLAYER_DIM), dtype=np.float32)
    mask   = np.ones(MAX_PLAYERS, dtype=bool)   # all padding by default

    players = frame_data.get('players', [])
    for i, player in enumerate(players[:MAX_PLAYERS]):
        loc = player.get('location')
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        tokens[i, 0] = float(loc[0]) / PITCH_W               # x_norm
        tokens[i, 1] = float(loc[1]) / PITCH_H               # y_norm
        tokens[i, 2] = float(player.get('teammate', False))   # is_teammate
        tokens[i, 3] = float(player.get('keeper',   False))   # is_keeper
        tokens[i, 4] = float(player.get('actor',    False))   # is_actor
        mask[i] = False                                        # valid player

    return tokens, mask


class DatasetBuilderV3:
    """
    Builds the xT_v3 dataset: only events with a 360 freeze frame are kept,
    and each event is accompanied by a padded player token array.
    """

    def __init__(self):
        os.makedirs(DATA_DIR_V3, exist_ok=True)
        sys.path.insert(0, _V2_DIR)   # ensure loader is importable
        self.loader = StatsBomb360Loader()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, limit: int | None = None, force_rebuild: bool = False) -> None:
        match_ids = self.loader.get_360_matches()

        if limit:
            match_ids = match_ids[:limit]
            print(f"Limited to first {limit} matches.\n")

        skipped  = 0
        failures = 0
        to_build = []

        for mid in match_ids:
            out_path = os.path.join(DATA_DIR_V3, f"{mid}.npz")
            if os.path.exists(out_path) and not force_rebuild:
                skipped += 1
            else:
                to_build.append((mid, out_path))

        print(f"Matches to build: {len(to_build)}  |  Already built (skipping): {skipped}\n")

        with ThreadPoolExecutor(max_workers=BUILD_WORKERS) as executor:
            futures = {
                executor.submit(self._process_and_save, mid, path): mid
                for mid, path in to_build
            }
            with tqdm(total=len(futures), desc="Building v3 dataset") as pbar:
                for future in as_completed(futures):
                    mid = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        tqdm.write(f"  [SKIP] Match {mid}: {e}")
                        failures += 1
                    pbar.update(1)

        print(f"\nBuild complete.  Skipped: {skipped}  |  Failures: {failures}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all .npz files and return concatenated arrays.

        Returns
        -------
        spatial        : (N, NUM_CHANNELS, GRID_H, GRID_W)
        scalar         : (N, SCALAR_DIM)
        player_tokens  : (N, MAX_PLAYERS, PLAYER_DIM)
        player_mask    : (N, MAX_PLAYERS) bool
        labels         : (N,)
        match_ids      : (N,) int
        """
        files = sorted(f for f in os.listdir(DATA_DIR_V3) if f.endswith('.npz'))

        if not files:
            raise FileNotFoundError(
                f"No .npz files found in {DATA_DIR_V3}.\n"
                "Run  python main.py --build  first."
            )

        spatials, scalars, tokens, masks, labels, match_ids = [], [], [], [], [], []

        for fname in tqdm(files, desc="Loading v3 dataset"):
            data = np.load(os.path.join(DATA_DIR_V3, fname))
            spatials.append(data['spatial'])
            scalars.append(data['scalar'])
            tokens.append(data['player_tokens'])
            masks.append(data['player_mask'])
            labels.append(data['label'])
            mid = int(data['match_id'][0])
            match_ids.extend([mid] * len(data['label']))

        return (
            np.concatenate(spatials, axis=0),
            np.concatenate(scalars,  axis=0),
            np.concatenate(tokens,   axis=0),
            np.concatenate(masks,    axis=0),
            np.concatenate(labels,   axis=0),
            np.array(match_ids),
        )

    # ------------------------------------------------------------------
    # Internal: process one match
    # ------------------------------------------------------------------

    def _process_and_save(self, match_id: int, out_path: str) -> None:
        parser = MatchParser360(match_id)
        df, frame_lookup = parser.parse()

        if df.empty or not frame_lookup:
            raise ValueError("No 360 freeze frame data — skipping match.")

        df = self._assign_chains(df)

        # Keep only events that have a freeze frame
        has_frame = df['id'].isin(frame_lookup)
        df = df[has_frame].reset_index(drop=True)

        if df.empty:
            raise ValueError("No events with freeze frames after filtering.")

        n = len(df)
        spatial_arr = np.zeros((n, NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
        scalar_arr  = np.zeros((n, len(SCALAR_COLS)),              dtype=np.float32)
        token_arr   = np.zeros((n, MAX_PLAYERS, PLAYER_DIM),       dtype=np.float32)
        mask_arr    = np.ones( (n, MAX_PLAYERS),                    dtype=bool)

        for i, (_, row) in enumerate(df.iterrows()):
            event_id   = row.get('id', None)
            frame_data = frame_lookup.get(event_id, None)

            spatial_arr[i] = encode_event(row.to_dict(), frame_data)

            for j, col in enumerate(SCALAR_COLS):
                val = row.get(col, 0.0)
                scalar_arr[i, j] = float(val) if not pd.isna(val) else 0.0

            if frame_data:
                token_arr[i], mask_arr[i] = _extract_player_tokens(frame_data)

        labels = df['chain_goal'].values.astype(np.float32)

        np.savez_compressed(
            out_path,
            spatial       = spatial_arr,
            scalar        = scalar_arr,
            player_tokens = token_arr,
            player_mask   = mask_arr,
            label         = labels,
            match_id      = np.array([match_id]),
        )

    # ------------------------------------------------------------------
    # Internal: possession chains (identical to v2)
    # ------------------------------------------------------------------

    def _assign_chains(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['match_id', 'period', 'index']).reset_index(drop=True)

        match_change  = df['match_id'] != df['match_id'].shift(1)
        period_change = df['period']   != df['period'].shift(1)
        prev_was_shot = df['type'].shift(1) == 'Shot'

        if 'team' in df.columns:
            team_change = df['team'] != df['team'].shift(1)
        elif 'team_id' in df.columns:
            team_change = df['team_id'] != df['team_id'].shift(1)
        else:
            team_change = pd.Series(False, index=df.index)

        is_new_chain   = match_change | period_change | team_change | prev_was_shot
        df['chain_id'] = is_new_chain.cumsum()

        is_goal     = (df['type'] == 'Shot') & (df['success'] == 1)
        goal_chains = set(df.loc[is_goal, 'chain_id'].unique())

        labels = np.zeros(len(df), dtype=np.float32)
        for chain_id, group in df.groupby('chain_id'):
            if chain_id not in goal_chains:
                continue
            n     = len(group)
            steps = np.arange(n - 1, -1, -1, dtype=np.float32)
            labels[group.index] = CHAIN_GAMMA ** steps

        df['chain_goal'] = labels
        return df
