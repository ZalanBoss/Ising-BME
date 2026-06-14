from data_extraction import *

spins = []
betas = []
hs = []

for d in dataset:
    spins.append(d["sigma"])
    betas.append(d["beta"])
    hs.append(d["h"])

def energy(sigma, h):
    H = -h * np.sum(sigma)
    H -= np.sum(sigma[:-1, :] * sigma[1:, :])
    H -= np.sum(sigma[:, :-1] * sigma[:, 1:])
    return H


def C(sigma, r):
      m = sigma.mean()
      return np.mean(sigma * np.roll(sigma, r, axis=0)) - m**2

def features(sigma, h):
    # The total average magnetization
    M = np.sum(sigma) / np.size(sigma)
    # Energy per spin
    E = energy(sigma, h) / np.size(sigma)
    # Correlation Functions for r = {1,2,5,10,20}
    Cs = [C(sigma,r) for r in [1, 2, 5, 10, 20]]

    return np.array([M, E, *Cs])

import umap
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

X = np.array([features(d["sigma"], d["h"]) for d in dataset])
X = StandardScaler().fit_transform(X)   # normalize before UMAP

embedding = umap.UMAP(n_components=2).fit_transform(X)  # 300×2

gmm = GaussianMixture(n_components=2).fit(embedding)
labels = gmm.predict(embedding)

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sc1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=betas, cmap='plasma', s=15) #type: ignore
plt.colorbar(sc1, ax=ax1, label='beta')
ax1.set_title("UMAP — colored by beta")
ax1.set_xlabel("UMAP 1")
ax1.set_ylabel("UMAP 2")

ax2.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='coolwarm', s=15) #type:ignore
ax2.set_title("UMAP — colored by GMM cluster")
ax2.set_xlabel("UMAP 1")
ax2.set_ylabel("UMAP 2")

plt.tight_layout()
plt.savefig("umap_embedding.png", dpi=150)
plt.show()

plt.figure()
plt.scatter(betas, labels, c=labels)
plt.axvline(0.4407, color='r', linestyle='--', label='True Tc')
plt.xlabel("beta")
plt.ylabel("cluster")
plt.legend()
plt.savefig("cluster_vs_beta.png", dpi=150)
plt.show()

