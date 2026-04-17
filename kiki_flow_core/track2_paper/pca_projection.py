"""2D PCA projection for Aeon embeddings visualization."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class PCAProjection:
    """Scikit-learn PCA wrapper with fit / project / inverse and deterministic seeding."""

    def __init__(self, n_components: int = 2, seed: int = 0) -> None:
        self.n_components = n_components
        self.seed = seed
        self._pca: PCA | None = None

    def fit(self, embeddings: np.ndarray) -> PCAProjection:
        self._pca = PCA(n_components=self.n_components, random_state=self.seed)
        self._pca.fit(embeddings)
        return self

    def project(self, x: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("Call .fit() first")
        out: np.ndarray = self._pca.transform(x)
        return out

    def inverse(self, projected: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("Call .fit() first")
        out: np.ndarray = self._pca.inverse_transform(projected)
        return out
