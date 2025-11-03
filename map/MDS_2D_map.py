import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

D = pd.read_csv("map/flight_distances.csv", index_col=0)

# MDS
X_labels = D.index.to_list()
n = len(D)
D2 = D.values**2
J  = np.eye(n) - np.ones((n, n))/n
A  = -0.5 * J @ D2 @ J

# eigendecomposition (keep top 2 positive eigenpairs)
w, V = eigh(A)                 # ascending order
idx = np.argsort(w)[::-1]      # descending
w, V = w[idx], V[:, idx]
pos = w > 1e-9
w, V = w[pos], V[:, pos]
r = min(2, len(w))
X = V[:, :r] * np.sqrt(w[:r])  # coordinates (n x r)
X = -X  # Make the map look familiar

# visualize
# shared bounds + aspect
pad = 100
xmin, ymin = X.min(axis=0) - pad
xmax, ymax = X.max(axis=0) + pad

def setup_axes(ax, title):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(X[:,0], X[:,1])
for i, name in enumerate(X_labels):
    ax.text(X[i,0]+0.01, X[i,1]+0.01, name, fontsize=8)
setup_axes(ax, "2D map of cities")
fig.tight_layout()
fig.savefig("map/figures/MDS_2D_map.png", bbox_inches="tight", dpi=300)


# Range of k values to evaluate
K = range(1, 10)
inertias = []

for k in K:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters k"); plt.ylabel("Inertia (within-cluster SSE)")
plt.xticks(K); plt.grid(True)
plt.savefig("map/figures/elbow_method.png")


# K-means clustering of cities
k = 4  # optimal number from elbow method
km = KMeans(n_clusters=k, n_init="auto", random_state=42)
clusters = km.fit_predict(X)

# visualize
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap="tab10", s=80)
for i, name in enumerate(X_labels):
    plt.text(X[i,0]+0.01, X[i,1]+0.01, name, fontsize=8)
plt.title(f"K-means clustering of cities (k={k})")
plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
plt.savefig("map/figures/kmeans_clustering.png")

fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.scatter(X[:,0], X[:,1], c=clusters, cmap="tab10", s=80)
for i, name in enumerate(X_labels):
    ax2.text(X[i,0]+0.01, X[i,1]+0.01, name, fontsize=8)
setup_axes(ax2, "K-means clustering of cities (k={k})")
fig2.tight_layout()
fig2.savefig("map/figures/kmeans_clustering.png", bbox_inches="tight", dpi=300)


# build Voronoi from cluster centers
centers = km.cluster_centers_
vor = Voronoi(centers)

# visualize
fig3, ax3 = plt.subplots(figsize=(8,6))
voronoi_plot_2d(vor, ax=ax3, show_vertices=False, line_width=1.0)
ax3.scatter(X[:,0], X[:,1], c=km.labels_, s=60)
ax3.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], marker='X', s=160, edgecolor='k')
for i, name in enumerate(X_labels):
    ax3.text(X[i,0]+0.01, X[i,1]+0.01, name, fontsize=8)
setup_axes(ax3, f"Voronoi over K-means centers (k={len(set(km.labels_))})")
fig3.tight_layout()
fig3.savefig("map/figures/kmeans_voronoi.png", bbox_inches="tight", dpi=300)
