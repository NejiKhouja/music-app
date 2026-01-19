import joblib
import pandas as pd
import numpy as np
import os

class Recommender:
    def __init__(self):
        self.ml_dir = r"C:\Users\deadx\OneDrive\Desktop\music-app\ml"
        self.scaler = None
        self.model = None
        self.features = None
        self.loaded = False
        self.load()
    
    def load(self):
        try:
            # Load ML model files
            self.scaler = joblib.load(os.path.join(self.ml_dir, "models/scaler.pkl"))
            self.model = joblib.load(os.path.join(self.ml_dir, "models/similarity_model.pkl"))
            
            # Load features
            features_df = pd.read_csv(os.path.join(self.ml_dir, "data/processed/features.csv"))
            self.features = features_df.select_dtypes(include='number')
            
            self.loaded = True
            print(f"✅ ML models loaded. {len(self.features)} tracks available.")
            return True
        except Exception as e:
            print(f"❌ Error loading ML: {e}")
            return False
    
    def get_recommendations(self, ml_index, n=10):
        if not self.loaded or ml_index >= len(self.features):
            return []
        
        # Get features and scale
        song_features = self.scaler.transform([self.features.iloc[ml_index]])
        
        # Find similar tracks
        max_n = min(100, len(self.features))
        distances, indices = self.model.kneighbors(song_features, n_neighbors=max_n)
        
        # Collect recommendations
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == ml_index:  # Skip self
                continue
            
            results.append({
                'ml_index': int(idx),
                'distance': float(dist)
            })
            
            if len(results) >= n:
                break
        
        return results

# Create global instance
recommender = Recommender()