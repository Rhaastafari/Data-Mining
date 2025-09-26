import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# CSV laden
df = pd.read_csv("energy_data.csv")

# Features (ohne "Country")
X = df.drop("Country", axis=1)

# Standardisieren
X_scaled = StandardScaler().fit_transform(X)

# PCA (wie vorher, z.B. 2 Komponenten)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Silhouette Scores berechnen für verschiedene k
silhouette_scores = []
K = range(2, 11)  # Silhouette Score ist ab k=2 definiert
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    silhouette_scores.append(score)

# Plot
plt.figure(figsize=(8,6))
plt.plot(K, silhouette_scores, marker='o', linestyle='--')
plt.title("Silhouette Scores for k-Means Clustering")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()

# Speichern
plt.savefig("silhouetteplot.pdf")
plt.savefig("silhouetteplot.png")
plt.show()
