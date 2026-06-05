import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Features
X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical Clustering
hc = AgglomerativeClustering(
    n_clusters=5,
    metric="euclidean",
    linkage="ward"
)

clusters = hc.fit_predict(X_scaled)

# Add cluster column
df["Cluster"] = clusters

# Streamlit UI
st.title("Hierarchical Clustering")

st.write("Customer Segmentation using Hierarchical Clustering")

st.dataframe(df.head())

income = st.number_input(
    "Annual Income (k$)",
    min_value=1,
    max_value=200,
    value=50
)

score = st.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)

if st.button("Predict Cluster"):

    point = np.array([[income, score]])

    point_scaled = scaler.transform(point)

    distances = np.linalg.norm(
        X_scaled - point_scaled,
        axis=1
    )

    idx = np.argmin(distances)

    predicted_cluster = clusters[idx]

    st.success(
        f"Customer belongs to Cluster {predicted_cluster}"
    )
