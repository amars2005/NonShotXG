import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DATA_DIR, SCALAR_COLS, NUM_CHANNELS, GRID_H, GRID_W, CHAIN_GAMMA, BUILD_WORKERS
from loader import StatsBomb360Loader
from parser import MatchParser360
from encoder import encode_event


class DatasetBuilder:
    """
    Orchestrates the full data pipeline:
      1. Discover all 360-compatible match IDs via the loader.
      2. Parse each match (events + freeze frames) via MatchParser360.
      3. Assign possession chains and goal labels.
      4. Encode each event into a spatial tensor + scalar vector.
      5. Save each match as a compressed .npz file to DATA_DIR.

    On subsequent runs, already-built .npz files are skipped unless
    force_rebuild=True is passed to build().
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.loader = StatsBomb360Loader()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, limit: int | None = None, force_rebuild: bool = False) -> None:
        """
        Build (or incrementally update) the encoded dataset on disk.

        Parameters
        ----------
        limit : int, optional
            Cap the number of matches processed — useful for quick testing.
        force_rebuild : bool
            Re-encode matches even if their .npz already exists.
        """
        match_ids = self.loader.get_360_matches()

        if limit:
            match_ids = match_ids[:limit]
            print(f"Limited to first {limit} matches.\n")

        skipped  = 0
        failures = 0

        to_build = []
        for match_id in match_ids:
            out_path = os.path.join(DATA_DIR, f"{match_id}.npz")
            if os.path.exists(out_path) and not force_rebuild:
                skipped += 1
            else:
                to_build.append((match_id, out_path))

        print(f"Matches to build: {len(to_build)}  |  Already built (skipping): {skipped}\n")

        with ThreadPoolExecutor(max_workers=BUILD_WORKERS) as executor:
            futures = {
                executor.submit(self._process_and_save, mid, path): mid
                for mid, path in to_build
            }
            with tqdm(total=len(futures), desc="Building dataset") as pbar:
                for future in as_completed(futures):
                    mid = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        tqdm.write(f"  [SKIP] Match {mid}: {e}")
                        failures += 1
                    pbar.update(1)

        print(f"\nBuild complete.  Skipped (already built): {skipped}  |  Failures: {failures}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load every .npz file from DATA_DIR and return concatenated arrays.

        Returns
        -------
        spatial   : (N, NUM_CHANNELS, GRID_H, GRID_W) float32
        scalar    : (N, SCALAR_DIM) float32
        labels    : (N,) float32  — chain_goal
        match_ids : (N,) int      — match ID for each event (used for train/val split)
        """
        files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.npz'))

        if not files:
            raise FileNotFoundError(
                f"No .npz files found in {DATA_DIR}.\n"
                "Run  python main.py --build  first."
            )

        spatials, scalars, labels, match_ids = [], [], [], []

        for fname in tqdm(files, desc="Loading dataset"):
            data = np.load(os.path.join(DATA_DIR, fname))
            spatials.append(data['spatial'])
            scalars.append(data['scalar'])
            labels.append(data['label'])
            mid = int(data['match_id'][0])
            match_ids.extend([mid] * len(data['label']))

        return (
            np.concatenate(spatials,  axis=0),
            np.concatenate(scalars,   axis=0),
            np.concatenate(labels,    axis=0),
            np.array(match_ids),
        )

    # ------------------------------------------------------------------
    # Internal: process one match
    # ------------------------------------------------------------------

    def _process_and_save(self, match_id: int, out_path: str) -> None:
        parser = MatchParser360(match_id)
        df, frame_lookup = parser.parse()

        if df.empty:
            return

        # Reject matches where the 360 JSON was unavailable (404 or network error).
        # These build without player channel data and would pollute the training set.
        if not frame_lookup:
            raise ValueError("No 360 freeze frame data returned — skipping match.")

        df = self._assign_chains(df)

        n = len(df)
        spatial_arr = np.zeros((n, NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
        scalar_arr  = np.zeros((n, len(SCALAR_COLS)),              dtype=np.float32)

        for i, (_, row) in enumerate(df.iterrows()):
            event_id   = row.get('id', None)
            frame_data = frame_lookup.get(event_id, None)

            spatial_arr[i] = encode_event(row.to_dict(), frame_data)

            for j, col in enumerate(SCALAR_COLS):
                val = row.get(col, 0.0)
                scalar_arr[i, j] = float(val) if not pd.isna(val) else 0.0

        labels = df['chain_goal'].values.astype(np.float32)

        np.savez_compressed(
            out_path,
            spatial  = spatial_arr,
            scalar   = scalar_arr,
            label    = labels,
            match_id = np.array([match_id]),
        )

    # ------------------------------------------------------------------
    # Internal: possession chains
    # ------------------------------------------------------------------

    def _assign_chains(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Segments events into possession chains and assigns a temporally
        discounted label to each event.

        For goal-scoring chains, each event receives:
            label = CHAIN_GAMMA ^ (steps_from_chain_end)

        So the shot that scores = 1.0, the pass before it = CHAIN_GAMMA,
        the event before that = CHAIN_GAMMA^2, etc. Events in non-goal
        chains receive label = 0.

        A new chain begins when:
          - The match changes
          - The period changes
          - The team in possession changes
          - The previous event was a Shot (possession always ends on a shot)
        """
        df = df.sort_values(['match_id', 'period', 'index']).reset_index(drop=True)

        match_change  = df['match_id'] != df['match_id'].shift(1)
        period_change = df['period']   != df['period'].shift(1)
        prev_was_shot = df['type'].shift(1) == 'Shot'

        # Determine team column
        if 'team' in df.columns:
            team_change = df['team'] != df['team'].shift(1)
        elif 'team_id' in df.columns:
            team_change = df['team_id'] != df['team_id'].shift(1)
        else:
            team_change = pd.Series(False, index=df.index)

        is_new_chain = match_change | period_change | team_change | prev_was_shot
        df['chain_id'] = is_new_chain.cumsum()

        # Identify goal chains
        is_goal     = (df['type'] == 'Shot') & (df['success'] == 1)
        goal_chains = set(df.loc[is_goal, 'chain_id'].unique())

        # Assign temporally discounted labels
        labels = np.zeros(len(df), dtype=np.float32)
        for chain_id, group in df.groupby('chain_id'):
            if chain_id not in goal_chains:
                continue
            n = len(group)
            # steps_from_end: last event = 0, second-to-last = 1, ...
            steps = np.arange(n - 1, -1, -1, dtype=np.float32)
            labels[group.index] = CHAIN_GAMMA ** steps

        df['chain_goal'] = labels
        return df
