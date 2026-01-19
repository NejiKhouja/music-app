# src/inference_spotify_fixed.py

import pandas as pd
import numpy as np
import joblib

features = pd.read_csv("../data/processed/features.csv")
numeric_features = features.select_dtypes(include='number')

tracks = pd.read_csv("../data/raw/SpotifyFeatures.csv")  

scaler = joblib.load("../models/scaler.pkl")
similarity_model = joblib.load("../models/similarity_model.pkl")

song_idx = 61659
n_neighbors = 20       
match_genre = False     

print(numeric_features.shape)
song_features = scaler.transform([numeric_features.iloc[song_idx]])

max_neighbors = min(100, numeric_features.shape[0])  
distances, indices = similarity_model.kneighbors(song_features, n_neighbors=max_neighbors)

reference_genre = tracks.iloc[song_idx]['genre']
recommendations = []

for idx, dist in zip(indices[0], distances[0]):
    if idx == song_idx:
        continue  
    if match_genre and tracks.iloc[idx]['genre'] != reference_genre:
        continue
    recommendations.append((idx, dist))
    if len(recommendations) >= n_neighbors:
        break

ref_song = tracks.iloc[song_idx]
print(f"Reference song: {ref_song['track_name']} - {ref_song['artist_name']} (genre: {reference_genre})\n")
print(f"Top {len(recommendations)} similar songs (match_genre={match_genre}):\n")

for rank, (idx, dist) in enumerate(recommendations, start=1):
    song = tracks.iloc[idx]
    print(f"{rank:02d}. {song['track_name']} - {song['artist_name']} "
          f"(distance: {dist:.4f}) genre: {song['genre']})")
