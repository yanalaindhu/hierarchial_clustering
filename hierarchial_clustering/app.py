import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

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

df["Cluster"] = clusters

# Streamlit UI
st.set_page_config(
    page_title="Hierarchical Clustering",
    page_icon="📊"
)

st.title("📊 Customer Segmentation using Hierarchical Clustering")

st.write("Mall Customer Segmentation Project")

if st.checkbox("Show Dataset"):
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

    nearest_index = np.argmin(distances)

    predicted_cluster = clusters[nearest_index]

    st.success(
        f"Customer belongs to Cluster {predicted_cluster}"
    )
