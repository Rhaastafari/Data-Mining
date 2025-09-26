import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# CSV laden
df = pd.read_csv("energy_data.csv")

# Features ohne "Country"
X = df.drop("Country", axis=1)

# Standardisieren
X_scaled = StandardScaler().fit_transform(X)

# PCA -> Reduktion auf 2 Komponenten
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans mit optimalem k (z.B. 3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_pca)

# Durchschnittswerte pro Cluster berechnen
cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
print(cluster_summary.round(2))
