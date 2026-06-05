import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("Mall_Customers.csv")

# -------------------------------
# Feature Selection
# -------------------------------

X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

# -------------------------------
# Feature Scaling
# -------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------------
# Hierarchical Clustering
# -------------------------------

hc = AgglomerativeClustering(
    n_clusters=5,
    metric="euclidean",
    linkage="ward"
)

clusters = hc.fit_predict(X_scaled)

df["Cluster"] = clusters

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(
    page_title="Hierarchical Clustering",
    page_icon="📊"
)

st.title("📊 Customer Segmentation Using Hierarchical Clustering")

st.write(
    "Segment customers based on Income and Spending Score"
)

# -------------------------------
# Dataset Preview
# -------------------------------

if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# -------------------------------
# Dendrogram
# -------------------------------

if st.checkbox("Show Dendrogram"):

    fig, ax = plt.subplots(figsize=(10, 5))

    dendrogram(
        linkage(
            X_scaled,
            method="ward"
        ),
        ax=ax
    )

    ax.set_title("Dendrogram")

    st.pyplot(fig)

# -------------------------------
# Cluster Visualization
# -------------------------------

if st.checkbox("Show Clusters"):

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Segments")

    st.pyplot(fig)

# -------------------------------
# Predict Cluster
# -------------------------------

st.subheader("Find Customer Cluster")

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

    predicted_cluster = clusters[
        nearest_index
    ]

    st.success(
        f"Customer belongs to Cluster {predicted_cluster}"
    )
