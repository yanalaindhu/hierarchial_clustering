import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Select features
X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

# Scale data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Train Hierarchical Clustering
hc = AgglomerativeClustering(
    n_clusters=5,
    metric="euclidean",
    linkage="ward"
)

clusters = hc.fit_predict(X_scaled)

# Add cluster labels
df["Cluster"] = clusters

# Save clustered dataset
df.to_csv(
    "models/clustered_customers.csv",
    index=False
)

# Save scaler
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Hierarchical Clustering Training Completed")
print("Files Saved:")
print("models/clustered_customers.csv")
print("models/scaler.pkl")