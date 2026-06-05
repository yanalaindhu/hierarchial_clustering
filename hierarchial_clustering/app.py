import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import cdist

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Hierarchical Clustering Dashboard",
    page_icon="🌳",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/Mall_Customers.csv")

scaler = pickle.load(open("models/scaler.pkl", "rb"))

cluster_centers = pickle.load(
    open("models/cluster_centers.pkl", "rb")
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp{
    background-color:#F4F7FC;
}

h1,h2,h3{
    color:#1F2937 !important;
}

p,label{
    color:#374151 !important;
}

.stButton > button{
    width:100%;
    height:3rem;
    border-radius:12px;
    font-size:18px;
    font-weight:600;

    background-color:#C7D2FE !important;   /* pastel lavender */
    color:#312E81 !important;
    border:none;
}

.stButton > button:hover{
    background-color:#A5B4FC !important;
    color:#1E1B4B !important;
}

.custom-card{
    padding:18px;
    border-radius:12px;
    margin-top:10px;
}

.success-card{
    background:#DCFCE7;
    color:#166534;
    font-weight:bold;
}

.info-card{
    background:#DBEAFE;
    color:#1E40AF;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("🌳 Hierarchical Clustering Dashboard")

st.markdown("""
Customer segmentation using
**Agglomerative Hierarchical Clustering**
""")

# ==========================================
# METRICS
# ==========================================

c1,c2 = st.columns(2)

with c1:
    st.metric("Total Customers", len(df))

with c2:
    st.metric("Clusters", len(cluster_centers))

st.divider()

# ==========================================
# INPUTS
# ==========================================

st.subheader("🔍 Predict Customer Segment")

col1,col2 = st.columns(2)

with col1:
    income = st.slider(
        "Annual Income (k$)",
        0,
        150,
        60
    )

with col2:
    score = st.slider(
        "Spending Score",
        1,
        100,
        50
    )

# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict Segment"):

    sample = np.array([[income, score]])

    distances = cdist(
        sample,
        cluster_centers.values,
        metric='euclidean'
    )

    cluster = np.argmin(distances)

    st.markdown(
        f"""
        <div class="custom-card success-card">
        Predicted Cluster : {cluster}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="custom-card info-card">
        Customer belongs to Segment {cluster}
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# SCATTER PLOT
# ==========================================

st.subheader("📈 Customer Segments")

clusters = []

for _, row in df.iterrows():

    point = np.array([
        [row['Annual Income (k$)'],
         row['Spending Score (1-100)']]
    ])

    d = cdist(
        point,
        cluster_centers.values,
        metric='euclidean'
    )

    clusters.append(np.argmin(d))

fig, ax = plt.subplots(figsize=(10,6))

scatter = ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=clusters,
    s=80
)

ax.scatter(
    income,
    score,
    color='red',
    marker='X',
    s=300,
    label='Selected Customer'
)

ax.set_title(
    "Hierarchical Customer Segmentation"
)

ax.set_xlabel(
    "Annual Income (k$)"
)

ax.set_ylabel(
    "Spending Score"
)

ax.legend()

st.pyplot(fig)

# ==========================================
# DENDROGRAM
# ==========================================

st.subheader("🌳 Dendrogram")

sample_df = df.sample(
    50,
    random_state=42
)

X = sample_df[
    ['Annual Income (k$)',
     'Spending Score (1-100)']
]

X_scaled = scaler.transform(X)

fig2, ax2 = plt.subplots(
    figsize=(12,5)
)

linked = linkage(
    X_scaled,
    method='ward'
)

dendrogram(
    linked,
    ax=ax2
)

ax2.set_title(
    "Hierarchical Clustering Dendrogram"
)

st.pyplot(fig2)

# ==========================================
# CLUSTER DISTRIBUTION
# ==========================================

st.subheader("📊 Cluster Distribution")

cluster_counts = (
    pd.Series(clusters)
    .value_counts()
    .sort_index()
)

fig3, ax3 = plt.subplots(
    figsize=(10,5)
)

bars = ax3.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

ax3.set_title(
    "Customers per Cluster"
)

ax3.set_xlabel(
    "Cluster"
)

ax3.set_ylabel(
    "Count"
)

for bar in bars:

    height = bar.get_height()

    ax3.text(
        bar.get_x()+bar.get_width()/2,
        height,
        str(height),
        ha='center'
    )

st.pyplot(fig3)
