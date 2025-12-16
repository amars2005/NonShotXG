import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt

class ExpectedThreatModel:
    def __init__(self, filepath="statsbomb_chained_dataset.csv"):
        self.filepath = filepath
        self.model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        
    def load_and_prep(self):
        print("Loading dataset...")
        df = pd.read_csv(self.filepath)
        
        df['dx'] = 120 - df['start_x']
        df['dy'] = 40 - df['start_y']
        
        df['dist_to_goal'] = np.sqrt(df['dx']**2 + df['dy']**2)
        
        df['angle_to_goal'] = np.arctan2(df['dy'], df['dx'])
        
        df_encoded = pd.get_dummies(df, columns=['type'], prefix='type')
        
        features = ['start_x', 'start_y', 'dist_to_goal', 'angle_to_goal', 'type_Pass', 'type_Carry']
        
        for col in ['type_Pass', 'type_Carry']:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        df_encoded[features] = df_encoded[features].fillna(0)
        
        return df_encoded, features

    def train(self):
        df, features = self.load_and_prep()
        
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
        
        probs = self.model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        
        print(f"\n--- Model Results ---")
        print(f"ROC AUC Score: {auc:.3f} (0.5 is random, 1.0 is perfect)")
        print(f"Brier Score: {brier:.3f} (Lower is better)")
        
        all_probs = self.model.predict_proba(df[features])[:, 1]
        df['pred_value'] = all_probs
        
        return df

    def visualize_value_map(self, df):
        """
        Creates a heatmap of the pitch showing the 'Value' of having the ball in each zone.
        """
        x_range = np.linspace(0, 120, 50)
        y_range = np.linspace(0, 80, 50)
        xx, yy = np.meshgrid(x_range, y_range)
        
        grid_points = pd.DataFrame({
            'start_x': xx.ravel(),
            'start_y': yy.ravel()
        })
        
        grid_points['dx'] = 120 - grid_points['start_x']
        grid_points['dy'] = 40 - grid_points['start_y']
        grid_points['dist_to_goal'] = np.sqrt(grid_points['dx']**2 + grid_points['dy']**2)
        grid_points['angle_to_goal'] = np.arctan2(grid_points['dy'], grid_points['dx'])
        
        grid_points['type_Pass'] = 1
        grid_points['type_Carry'] = 0
        
        features = ['start_x', 'start_y', 'dist_to_goal', 'angle_to_goal', 'type_Pass', 'type_Carry']
        z = self.model.predict_proba(grid_points[features])[:, 1]
        z = z.reshape(xx.shape)
        
        print("Generating Heatmap...")
        plt.figure(figsize=(10, 7))
        plt.contourf(xx, yy, z, levels=20, cmap='magma')
        plt.colorbar(label='Probability of Goal (xT)')
        plt.title('Non-Shot Expected Goals Map (Random Forest)')
        plt.xlabel('Pitch Length (x)')
        plt.ylabel('Pitch Width (y)')
        
        plt.savefig("xt_heatmap.png")
        print("Heatmap saved to xt_heatmap.png")

if __name__ == "__main__":
    xt_model = ExpectedThreatModel()
    
    df_with_preds = xt_model.train()
    
    xt_model.visualize_value_map(df_with_preds)
    
    df_with_preds.to_csv("final_model_output.csv", index=False)