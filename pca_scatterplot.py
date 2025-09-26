import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# CSV laden
df = pd.read_csv("energy_data.csv")

# Features ohne "Country"
X = df.drop("Country", axis=1)

# Standardisieren
X_scaled = StandardScaler().fit_transform(X)

# PCA auf 2 Komponenten
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# K-Means mit optimalem k (z.B. 3 aus Elbow/Silhouette)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca)

# Scatterplot
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', s=80, alpha=0.8, edgecolors='k')

# Zentroiden einzeichnen
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, marker='X', label='Centroids')

# Achsen & Titel
plt.title("PCA Scatterplot with k-Means Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Speichern
plt.savefig("pca_scatterplot.pdf")
plt.savefig("pca_scatterplot.png")
plt.show()