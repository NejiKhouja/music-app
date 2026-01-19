# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load data and models once at startup
features = pd.read_csv("data/processed/features.csv")
numeric_features = features.select_dtypes(include='number')

tracks = pd.read_csv("data/raw/SpotifyFeatures.csv")

scaler = joblib.load("models/scaler.pkl")
similarity_model = joblib.load("models/similarity_model.pkl")

app = FastAPI()

class RecommendRequest(BaseModel):
    song_idx: int
    n_neighbors: int = 20
    match_genre: bool = False

@app.post("/recommend")
def recommend(req: RecommendRequest):
    song_idx = req.song_idx
    n_neighbors = req.n_neighbors
    match_genre = req.match_genre

    # Transform features
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
        recommendations.append({
            "idx": int(idx),
            "track_name": tracks.iloc[idx]['track_name'],
            "artist_name": tracks.iloc[idx]['artist_name'],
            "genre": tracks.iloc[idx]['genre'],
            "distance": float(dist)
        })
        if len(recommendations) >= n_neighbors:
            break

    ref_song = tracks.iloc[song_idx]
    return {
        "reference_song": {
            "track_name": ref_song['track_name'],
            "artist_name": ref_song['artist_name'],
            "genre": ref_song['genre']
        },
        "recommendations": recommendations
    }
