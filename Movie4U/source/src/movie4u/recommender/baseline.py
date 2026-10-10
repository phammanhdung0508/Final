"""KG-only scoring via User -> liked Movie -> Genre <- candidate Movie."""

import numpy as np


class KGOnlyRecommender:
    def __init__(self, dataset, popularity_weight=0.05):
        self.dataset = dataset
        features = dataset.features
        idf = np.log((1 + len(features)) / (1 + features.sum(axis=0))) + 1
        weighted = features * idf
        self.movie_vectors = weighted / np.maximum(
            np.linalg.norm(weighted, axis=1, keepdims=True), 1e-12
        )
        self.profiles = np.zeros(
            (len(dataset.users), features.shape[1]), dtype=np.float32
        )
        popularity = np.zeros(len(features), dtype=np.float32)
        for u, m in dataset.positive_train:
            self.profiles[u] += self.movie_vectors[m]
            popularity[m] += 1
        self.profiles /= np.maximum(
            np.linalg.norm(self.profiles, axis=1, keepdims=True), 1e-12
        )
        self.popularity = np.log1p(popularity) / max(
            float(np.log1p(popularity).max()), 1
        )
        self.popularity_weight = popularity_weight

    def scores(self):
        return (
            self.profiles @ self.movie_vectors.T
            + self.popularity_weight * self.popularity
        )
