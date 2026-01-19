# src/training2.py
# DBSCAN-based vibe clustering (no KMeans)

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import joblib

def load_features():
    return pd.read_csv(
        "/home/neji/python/music_recommender/data/processed/features.csv"
    )

# Training pipeline
def training_pipeline(
    eps=0.5,
    min_samples=10
):
    features = load_features()
    numeric_features = features.select_dtypes(include=np.number)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_features)

    model_dir = "/home/neji/python/music_recommender/models"
    os.makedirs(model_dir, exist_ok=True)

    # Save scaler + feature info
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    joblib.dump(
        {"numeric_features": list(numeric_features.columns)},
        f"{model_dir}/feature_info.pkl"
    )

    # DBSCAN clustering
    print(f"Training DBSCAN (eps={eps}, min_samples={min_samples})...")
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean"
    )
    labels = dbscan.fit_predict(X_scaled)

    # Save model + labels
    joblib.dump(dbscan, f"{model_dir}/dbscan.pkl")
    np.save(f"{model_dir}/dbscan_labels.npy", labels)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise}")

    similarity_model = NearestNeighbors(
        n_neighbors=20,
        metric="cosine"
    )
    similarity_model.fit(X_scaled)
    joblib.dump(
        similarity_model,
        f"{model_dir}/similarity_model.pkl"
    )

    print("DBSCAN training complete!")

    return {
        "features": features,
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise
    }


if __name__ == "__main__":
    training_pipeline(
        eps=0.7,      
        min_samples=15 
    )
