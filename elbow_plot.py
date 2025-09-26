import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# CSV laden (deine energy_data.csv)
df = pd.read_csv("energy_data.csv")

# Features extrahieren (ohne "Country")
X = df.drop("Country", axis=1)

# Standardisieren
X_scaled = StandardScaler().fit_transform(X)

# PCA auf 2 Komponenten (wie in deinem Paper)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Elbow Methode: WCSS für verschiedene k
wcss = []
K = range(1, 11)  # von 1 bis 10 Cluster
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

# Plot
plt.figure(figsize=(8,6))
plt.plot(K, wcss, marker='o', linestyle='--')
plt.title("Elbow Plot for k-Means Clustering")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.grid(True)
plt.tight_layout()

# Speichern
plt.savefig("elbowplot.pdf")
plt.savefig("elbowplot.png")
plt.show()