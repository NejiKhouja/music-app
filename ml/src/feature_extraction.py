# ml/src/preprocess.py (COMPLETE WORKING VERSION)
import pandas as pd
import os
import hashlib

def generate_track_id(track_name: str, artist_name: str) -> str:
    """Generate unique ID from track and artist"""
    unique_string = f"{track_name}_{artist_name}".lower().strip()
    return hashlib.md5(unique_string.encode()).hexdigest()[:16]

def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset"""
    return pd.read_csv(path)

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate tracks based on track name and artist"""
    return df.drop_duplicates(subset=['track_name', 'artist_name'], keep='first')

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create numerical features for ML"""
    df = df.copy()
    if {'energy', 'danceability'}.issubset(df.columns):
        df['energy_dance_combo'] = df['energy'] * df['danceability']
    if 'duration_ms' in df.columns:
        df['duration_min'] = df['duration_ms'] / 60000
    return df

def save_features(df: pd.DataFrame, save_path: str):
    """Save numerical features with track_id"""
    feature_cols = [
        'track_id',  # ← ADD THIS
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'energy_dance_combo', 'duration_min'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    features = df[feature_cols].copy()

    # Fill missing values
    for col in features.columns:
        if col != 'track_id' and features[col].isnull().any():
            features[col] = features[col].fillna(features[col].median())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    features.to_csv(save_path, index=False)
    print(f"Numerical features saved to {save_path}")

def save_metadata(df: pd.DataFrame, save_path: str):
    """Save metadata with track_id"""
    metadata_cols = ['track_id', 'track_name', 'artist_name', 'genre']  # ← ADD track_id
    metadata_cols = [c for c in metadata_cols if c in df.columns]
    metadata = df[metadata_cols].copy()

    # Fill missing values
    metadata = metadata.fillna('')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metadata.to_csv(save_path, index=False)
    print(f"Metadata saved to {save_path}")

def run_pipeline(data_path: str):
    # Load raw dataset
    print("Loading data...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} tracks")

    # Remove duplicates
    print("Removing duplicates...")
    df = remove_duplicates(df)
    print(f"After deduplication: {len(df)} tracks")

    # Generate unique track IDs
    print("Generating track IDs...")
    df['track_id'] = df.apply(
        lambda row: generate_track_id(row['track_name'], row['artist_name']), 
        axis=1
    )
    print(f"Generated {len(df['track_id'].unique())} unique track IDs")

    # Save metadata with track_id
    print("\nSaving metadata...")
    save_metadata(df, os.path.join("metadata", "tracks_metadata.csv"))

    # Create numerical features and save WITH SAME track_id
    print("Creating features...")
    df_features = create_features(df)
    save_features(df_features, os.path.join("features", "features.csv"))
    
    print("\n✅ Pipeline complete!")
    print(f"Metadata saved to: metadata/tracks_metadata.csv")
    print(f"Features saved to: features/features.csv")
    
    # Show sample
    print("\nSample track IDs:")
    for i, (track_id, name, artist) in enumerate(zip(df['track_id'].head(3), 
                                                     df['track_name'].head(3), 
                                                     df['artist_name'].head(3))):
        print(f"  {i+1}. {track_id} -> '{name}' by {artist}")

if __name__ == "__main__":
    # Test with a small sample first
    test_data = "../data/raw/SpotifyFeatures.csv"  # Adjust path
    
    # Create test directories if they don't exist
    os.makedirs("metadata", exist_ok=True)
    os.makedirs("features", exist_ok=True)
    
    print("=" * 50)
    print("Running ML Preprocessing Pipeline")
    print("=" * 50)
    
    # Test with first 100 rows if you want to test quickly
    # Uncomment for quick test:
    # df_test = pd.read_csv(test_data, nrows=100)
    # df_test.to_csv("test_sample.csv", index=False)
    # run_pipeline("test_sample.csv")
    
    # Run full pipeline
    run_pipeline(test_data)