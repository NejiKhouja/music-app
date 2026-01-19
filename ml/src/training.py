# src/training_vibe_aware_spotify.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import joblib


def load_features():
    return pd.read_csv("/home/neji/python/music_recommender/data/processed/features.csv")
def load_tracks():
    return pd.read_csv("/home/neji/python/music_recommender/data/raw/SpotifyFeatures.csv")
def cluster_analysis_plot(X_scaled, k_list, save_path="cluster_analysis.png"):
    inertia = []
    silhouette = []

    sample_size = 5000
    if X_scaled.shape[0] > sample_size:
        sample_idx = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
        X_sample = X_scaled[sample_idx]
    else:
        X_sample = X_scaled
        sample_idx = np.arange(X_scaled.shape[0])

    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(X_sample, labels[sample_idx]))

    best_k = k_list[np.argmax(silhouette)]

    plt.figure(figsize=(10,5))
    plt.plot(k_list, inertia, marker='o', label="Inertia (Elbow)")
    plt.plot(k_list, silhouette, marker='s', color='orange', label="Silhouette Score")
    plt.axvline(best_k, color='green', linestyle='--', label=f"Best k={best_k}")
    plt.title("KMeans Cluster Analysis")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Cluster analysis plot saved to {save_path}")
    return best_k

def training_pipeline(k_list=[2], plot_analysis=True):
    features = load_features()
    numeric_features = features.select_dtypes(include=np.number)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_features)

    model_dir = "/home/neji/python/music_recommender/models"
    os.makedirs(model_dir, exist_ok=True)

    # Save scaler and feature info
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    joblib.dump({"numeric_features": list(numeric_features.columns)}, f"{model_dir}/feature_info.pkl")

    #plot cluster analysis
    best_k = None
    if plot_analysis:
        best_k = cluster_analysis_plot(X_scaled, k_list, save_path=os.path.join(model_dir, "cluster_analysis3.png"))
        print(f"Best k according to silhouette score: {best_k}")

    results_all = {}
    for k in k_list:
        print(f"Training KMeans with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        joblib.dump(kmeans, f"{model_dir}/kmeans_k{k}.pkl")
        np.save(f"{model_dir}/cluster_labels_k{k}.npy", labels)

        results_all[k] = {"kmeans": kmeans, "labels": labels}

    similarity_model = NearestNeighbors(n_neighbors=20, metric='cosine')
    similarity_model.fit(X_scaled)
    joblib.dump(similarity_model, f"{model_dir}/similarity_model.pkl")

    print("Training complete!")
    return {"features": features,"results_all": results_all,"best_k": best_k}

if __name__ == "__main__":
    k_values = [2]
    results = training_pipeline(k_list=k_values, plot_analysis=False)
