import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

# 1) Daten laden
df = pd.read_csv("energy_data.csv")

# 2) Features vorbereiten (ohne Land)
X = df.drop(columns=["Country"])

# 3) Standardisieren + PCA(2)
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4) k-Means (k=3 – aus Elbow/Silhouette)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca)
df["Cluster"] = labels

# 5) Länder → ISO3 (für Plotly Choropleth)
iso3_map = {
    "Germany": "DEU",
    "France": "FRA",
    "USA": "USA",
    "China": "CHN",
    "India": "IND",
    "Brazil": "BRA",
    "Norway": "NOR",
    "Saudi Arabia": "SAU",
    "South Africa": "ZAF",
    "Japan": "JPN",
    "Canada": "CAN",
    "Australia": "AUS",
}
df["iso3"] = df["Country"].map(iso3_map)

# (Optional) prüfen, ob alle ISO-Codes da sind
if df["iso3"].isna().any():
    missing = df.loc[df["iso3"].isna(), "Country"].tolist()
    print("Warnung: fehlende ISO3-Codes für:", missing)

# 6) Interaktive Weltkarte: Clusterfarben pro Land
title = "Global Energy Consumption Clusters (PCA + k-Means)"
fig = px.choropleth(
    df,
    locations="iso3",
    color="Cluster",
    hover_name="Country",
    projection="natural earth",
    title=title
)
# Zusatzinfos im Hover anzeigen
fig.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>" +
                  "Cluster: %{z}<br>" +
                  "Energy use (GJ/cap): %{customdata[0]}<br>" +
                  "Renewables (%): %{customdata[1]}<br>" +
                  "CO2 (t/cap): %{customdata[2]}<extra></extra>",
    customdata=df[["EnergyUse_GJ_per_capita",
                   "Renewables_share_percent",
                   "CO2_tonnes_per_capita"]]
)

fig.update_layout(
    legend_title_text="Cluster",
    margin=dict(l=10, r=10, t=60, b=10)
)

# 7) Speichern
fig.write_html("cluster_map.html")       # interaktiv
fig.write_image("cluster_map.png", scale=2)  # benötigt 'kaleido'

print("Fertig: 'cluster_map.html' (interaktiv) und 'cluster_map.png' (statisch) gespeichert.")
