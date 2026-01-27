# model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Required for drawing circles/boxes

class ExpectedThreatModel:
    def __init__(self, filepath="statsbomb_chained_dataset.csv"):
        self.filepath = filepath
        # Increased depth slightly to capture more complex spatial patterns
        self.model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
        
    def load_and_prep(self):
        print("Loading dataset...")
        df = pd.read_csv(self.filepath)
        
        # 1. Feature Engineering
        # StatsBomb Goal location is (120, 40)
        df['dx'] = 120 - df['start_x']
        df['dy'] = 40 - df['start_y']
        df['dist_to_goal'] = np.sqrt(df['dx']**2 + df['dy']**2)
        df['angle_to_goal'] = np.arctan2(df['dy'], df['dx'])
        
        # 2. One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=['type'], prefix='type')
        
        features = ['start_x', 'start_y', 'dist_to_goal', 'angle_to_goal', 'type_Pass', 'type_Carry']
        
        # Ensure columns exist
        for col in ['type_Pass', 'type_Carry']:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        df_encoded[features] = df_encoded[features].fillna(0)
        
        return df_encoded, features

    def train(self):
        df, features = self.load_and_prep()
        
        # Split by Match ID
        match_ids = df['match_id'].unique()
        train_size = int(len(match_ids) * 0.8)
        train_matches = match_ids[:train_size]
        test_matches = match_ids[train_size:]
        
        print(f"Training on {len(train_matches)} matches, Testing on {len(test_matches)} matches.")
        
        train_data = df[df['match_id'].isin(train_matches)]
        test_data = df[df['match_id'].isin(test_matches)]
        
        X_train = train_data[features]
        y_train = train_data['chain_goal']
        X_test = test_data[features]
        y_test = test_data['chain_goal']
        
        print("Training Random Forest...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        probs = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        
        print(f"\n--- Model Results ---")
        print(f"ROC AUC Score: {auc:.3f}")
        print(f"Brier Score: {brier:.3f}")
        
        # Apply to whole dataset
        all_probs = self.model.predict_proba(df[features])[:, 1]
        df['pred_value'] = all_probs
        
        return df

    def _draw_pitch_markings(self, ax):
        """
        Helper function to draw standard football pitch markings 
        on StatsBomb coordinates (120x80).
        """
        lc = 'white' # Line color
        lw = 1.5     # Line width

        # 1. Outline & Center Line
        ax.plot([0, 0], [0, 80], color=lc, linewidth=lw)       # Left goal line
        ax.plot([120, 120], [0, 80], color=lc, linewidth=lw)   # Right goal line
        ax.plot([0, 120], [0, 0], color=lc, linewidth=lw)      # Bottom touchline
        ax.plot([0, 120], [80, 80], color=lc, linewidth=lw)    # Top touchline
        ax.plot([60, 60], [0, 80], color=lc, linewidth=lw)     # Halfway line

        # 2. Center Circle & Spot (Radius ~10 units)
        circle = mpatches.Circle((60, 40), 10, color=lc, fill=False, linewidth=lw)
        ax.add_patch(circle)
        ax.scatter(60, 40, color=lc, s=15) # Center spot

        # 3. Penalty Areas (18 yard box)
        # Left (y spans from 18 to 62, x spans 0 to 18)
        ax.plot([18, 18], [18, 62], color=lc, linewidth=lw)
        ax.plot([0, 18], [18, 18], color=lc, linewidth=lw)
        ax.plot([0, 18], [62, 62], color=lc, linewidth=lw)
        # Right (x spans 102 to 120)
        ax.plot([102, 102], [18, 62], color=lc, linewidth=lw)
        ax.plot([120, 102], [18, 18], color=lc, linewidth=lw)
        ax.plot([120, 102], [62, 62], color=lc, linewidth=lw)

        # 4. 6-Yard Boxes (Goal Areas)
        # Left (y spans 30 to 50, x spans 0 to 6)
        ax.plot([6, 6], [30, 50], color=lc, linewidth=lw)
        ax.plot([0, 6], [30, 30], color=lc, linewidth=lw)
        ax.plot([0, 6], [50, 50], color=lc, linewidth=lw)
        # Right (x spans 114 to 120)
        ax.plot([114, 114], [30, 50], color=lc, linewidth=lw)
        ax.plot([120, 114], [30, 30], color=lc, linewidth=lw)
        ax.plot([120, 114], [50, 50], color=lc, linewidth=lw)


    def visualize_value_map(self, df):
        """
        Creates a heatmap of the pitch showing the 'Value' of having the ball in each zone.
        """
        print("Generating Heatmap...")
        
        # Create grid
        x_range = np.linspace(0, 120, 50)
        y_range = np.linspace(0, 80, 50)
        xx, yy = np.meshgrid(x_range, y_range)
        
        grid_points = pd.DataFrame({'start_x': xx.ravel(), 'start_y': yy.ravel()})
        
        # Engineer grid features
        grid_points['dx'] = 120 - grid_points['start_x']
        grid_points['dy'] = 40 - grid_points['start_y']
        grid_points['dist_to_goal'] = np.sqrt(grid_points['dx']**2 + grid_points['dy']**2)
        grid_points['angle_to_goal'] = np.arctan2(grid_points['dy'], grid_points['dx'])
        # Assume Passes for the visualization
        grid_points['type_Pass'] = 1
        grid_points['type_Carry'] = 0
        
        features = ['start_x', 'start_y', 'dist_to_goal', 'angle_to_goal', 'type_Pass', 'type_Carry']
        z = self.model.predict_proba(grid_points[features])[:, 1]
        z = z.reshape(xx.shape)
        
        # --- PLOTTING UPDATED ---
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 1. Draw the heatmap first
        # Using 'magma' colormap: black/purple = low value, orange/bright = high value
        contour = ax.contourf(xx, yy, z, levels=25, cmap='magma', vmin=0, vmax=0.15)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Probability of Goal (xT)')
        
        # 2. Overlay pitch markings
        self._draw_pitch_markings(ax)
        
        # 3. Final Touches
        ax.set_title('Non-Shot Expected Goals Map (Random Forest)')
        ax.set_xlim(0, 120)
        ax.set_ylim(80, 0) # Invert Y axis so top-left is (0,0) - standard football view
        ax.set_aspect('equal') # Ensure the pitch isn't stretched
        ax.axis('off') # Hide the axis numbers for a cleaner look
        
        plt.tight_layout()
        plt.savefig("xt_heatmap.png", dpi=300, bbox_inches='tight')
        print("Heatmap saved to xt_heatmap.png")

# --- EXECUTION ---
if __name__ == "__main__":
    # NOTE: Ensure statsbomb_chained_dataset.csv exists first!
    try:
        xt_model = ExpectedThreatModel()
        df_with_preds = xt_model.train()
        xt_model.visualize_value_map(df_with_preds)
        df_with_preds.to_csv("final_model_output.csv", index=False)
    except FileNotFoundError:
        print("Error: 'statsbomb_chained_dataset.csv' not found.")
        print("Please run 'python main.py' first to generate the data.")