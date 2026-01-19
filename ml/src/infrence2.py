# src/inference2.py

import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load data
# ----------------------------
features = pd.read_csv(
    "/home/neji/python/music_recommender/data/processed/features.csv"
)

tracks = pd.read_csv(
    "/home/neji/python/music_recommender/data/raw/SpotifyFeatures.csv"
)

numeric_features = features.select_dtypes(include='number')

scaler = joblib.load(
    "/home/neji/python/music_recommender/models/scaler.pkl"
)

similarity_model = joblib.load(
    "/home/neji/python/music_recommender/models/similarity_model.pkl"
)

dbscan = joblib.load(
    "/home/neji/python/music_recommender/models/dbscan.pkl"
)

labels = np.load(
    "/home/neji/python/music_recommender/models/dbscan_labels.npy"
)

# ----------------------------
# Parameters
# ----------------------------
song_idx = 61659          # change this
n_recommendations = 20
use_cluster = False        # True = restrict to same DBSCAN cluster

# ----------------------------
# Prepare song features
# ----------------------------
song_features = scaler.transform(
    numeric_features.iloc[[song_idx]]
)

# ----------------------------
# Find neighbors
# ----------------------------
n_total = len(features)
n_neighbors = min(n_total, 5000)  # SAFE upper bound

distances, indices = similarity_model.kneighbors(
    song_features,
    n_neighbors=n_neighbors
)

# ----------------------------
# Filter recommendations
# ----------------------------
reference_cluster = labels[song_idx]
recommendations = []

for idx, dist in zip(indices[0], distances[0]):

    if idx == song_idx:
        continue

    # If cluster filtering enabled
    if use_cluster:
        # If reference is noise, allow all
        if reference_cluster != -1 and labels[idx] != reference_cluster:
            continue

    recommendations.append((idx, dist))

    if len(recommendations) >= n_recommendations:
        break

# ----------------------------
# Display results
# ----------------------------
ref_song = tracks.iloc[song_idx]

print(
    f"Reference song: {ref_song['track_name']} - "
    f"{ref_song['artist_name']} "
    f"(genre: {ref_song['genre']}, cluster: {reference_cluster})\n"
)

print(
    f"Top {len(recommendations)} similar songs "
    f"(use_cluster={use_cluster}):\n"
)

for rank, (idx, dist) in enumerate(recommendations, start=1):
    song = tracks.iloc[idx]
    print(
        f"{rank:02d}. {song['track_name']} - {song['artist_name']} "
        f"(distance: {dist:.4f}, "
        f"genre: {song['genre']}, "
        f"cluster: {labels[idx]})"
    )
