import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

# ----------------------------
# Load features
# ----------------------------
features = pd.read_csv("C:\Users\deadx\OneDrive\Desktop\music-app\ml\data\processed\features.csv")
numeric_features = features.select_dtypes(include=[np.number])

# ----------------------------
# Load scaler and scale data
# ----------------------------
scaler = joblib.load(
    "C:\Users\deadx\OneDrive\Desktop\music-app\ml\models\scaler.pkl"
)
X_scaled = scaler.transform(numeric_features)

print("Feature matrix shape:", X_scaled.shape)

# ----------------------------
# Load DBSCAN model and labels
# ----------------------------
dbscan = joblib.load(
    "C:\Users\deadx\OneDrive\Desktop\music-app\ml\models\dbscan.pkl"
)
labels = np.load(
    "C:\Users\deadx\OneDrive\Desktop\music-app\ml\models\dbscan_labels.npy")

# ----------------------------
# PCA to 2D
# ----------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(9, 7))

unique_labels = np.unique(labels)

for label in unique_labels:
    if label == -1:
        # Noise points
        plt.scatter(
            X_pca[labels == label, 0],
            X_pca[labels == label, 1],
            s=8,
            alpha=0.3,
            label="Noise",
            marker="x"
        )
    else:
        plt.scatter(
            X_pca[labels == label, 0],
            X_pca[labels == label, 1],
            s=10,
            alpha=0.6,
            label=f"Cluster {label}"
        )

plt.title("DBSCAN Clusters Visualization (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(markerscale=2)
plt.tight_layout()
plt.show()
