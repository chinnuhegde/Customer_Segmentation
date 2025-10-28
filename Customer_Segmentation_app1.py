import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("🧩 Customer Segmentation using K-Means Clustering")


# -------------------- Load Built-in Dataset --------------------
@st.cache_data
def load_dataset():
    # Change filename if you rename your file
    df = pd.read_csv("Mall_Customers.csv")
    return df

try:
    df = load_dataset()
except FileNotFoundError:
    st.error("❌ 'Mall_Customers.csv' not found! Please make sure it’s in the same folder as this script.")
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------- Data Preprocessing --------------------
st.subheader("⚙️ Data Preprocessing")

# Drop CustomerID if present
if "CustomerID" in df.columns:
    df.drop("CustomerID", axis=1, inplace=True)

# Encode Gender
if "Gender" in df.columns:
    st.write("Encoding Gender column...")
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    st.success("✅ Gender column encoded successfully!")

# Keep numeric columns
df_numeric = df.select_dtypes(include=[np.number])
st.write("✅ Columns used for clustering:")
st.write(df_numeric.columns.tolist())

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)

# -------------------- Elbow Method --------------------
st.subheader("📈 Elbow Method to Find Optimal Clusters")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker="o")
ax.set_title("Elbow Method")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("WCSS")
st.pyplot(fig)

# -------------------- Cluster Selection --------------------
st.subheader("🔢 Select Number of Clusters")
num_clusters = st.slider("Choose k (number of clusters)", 2, 10, 5)

kmeans = KMeans(n_clusters=num_clusters, init="k-means++", random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)
df["Cluster"] = cluster_labels

st.success(f"✅ Clustering complete! {num_clusters} clusters formed.")
st.dataframe(df.head())

# -------------------- Visualization --------------------
st.subheader("🎨 Cluster Visualization (2D)")
if df_numeric.shape[1] >= 2:
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df_numeric.iloc[:, 1],
        y=df_numeric.iloc[:, 2],
        hue=df["Cluster"],
        palette="Set1",
        s=80,
        ax=ax,
    )
    ax.set_xlabel(df_numeric.columns[1])
    ax.set_ylabel(df_numeric.columns[2])
    ax.set_title("Customer Clusters (2D View)")
    st.pyplot(fig)
else:
    st.warning("⚠️ Not enough numeric features for 2D visualization.")

# -------------------- Cluster Summary --------------------
st.subheader("📋 Cluster Summary")
summary = df.groupby("Cluster").mean(numeric_only=True)
st.dataframe(summary)

# -------------------- Download Results --------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Clustered Data as CSV",
    data=csv,
    file_name="clustered_customers.csv",
    mime="text/csv",
)
