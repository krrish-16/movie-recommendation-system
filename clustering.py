"""
clustering.py
-------------
Groups users by their movie-rating behaviour using K-Means clustering.

How it works
------------
1. Represent each user as a vector of ratings (the user-movie matrix row).
2. Run K-Means to partition users into K clusters.
3. Identify the most-loved movies in each cluster (highest avg rating).
4. Report cluster sizes and characteristic movies.

VIVA EXPLANATION:
  K-Means is an unsupervised learning algorithm that partitions N data points
  into K clusters by minimising within-cluster variance (inertia).
  In the context of recommender systems, clustering lets us group users with
  similar tastes and can be used for *group recommendations* or for improving
  CF by restricting neighbour search to a user's cluster.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD


class UserClusterer:
    """
    Cluster users in the user-movie matrix using K-Means.

    Parameters
    ----------
    n_clusters   : int   Number of clusters (default 5).
    random_state : int   For reproducibility.
    reduce_dims  : int   SVD components before clustering; 0 = no reduction.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        reduce_dims: int = 50,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.reduce_dims = reduce_dims

        self.matrix: pd.DataFrame | None = None
        self.labels: np.ndarray | None = None   # cluster label per user
        self.kmeans: KMeans | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Fit
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, user_movie_matrix: pd.DataFrame) -> None:
        """
        Fit K-Means on (optionally SVD-reduced) normalised user vectors.

        Parameters
        ----------
        user_movie_matrix : DataFrame
            Rows = users, columns = movie titles, values = ratings.
        """
        self.matrix = user_movie_matrix
        X = user_movie_matrix.values.astype(float)

        # Optional dimensionality reduction — speeds up K-Means on wide matrices
        n_components = min(self.reduce_dims, X.shape[1] - 1, X.shape[0] - 1)
        if self.reduce_dims > 0 and n_components > 1:
            print(f"[Clustering] Reducing dimensions to {n_components} via TruncatedSVD …")
            svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            X = svd.fit_transform(X)

        # L2-normalise so cosine distance ≈ Euclidean distance
        X = normalize(X, norm="l2")

        print(f"[Clustering] Fitting K-Means with K={self.n_clusters} …")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        self.labels = self.kmeans.fit_predict(X)
        print(f"[Clustering] Done.  Inertia: {self.kmeans.inertia_:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis helpers
    # ─────────────────────────────────────────────────────────────────────────

    def cluster_summary(self, top_movies: int = 5) -> pd.DataFrame:
        """
        Return a summary DataFrame: one row per cluster showing size and
        the most popular movies (highest average rating within that cluster).

        Parameters
        ----------
        top_movies : int   Number of representative movies per cluster.

        Returns
        -------
        DataFrame with columns [cluster, size, top_movies]
        """
        self._assert_fitted()
        rows = []

        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            cluster_users = self.matrix[mask]

            size = mask.sum()

            # Mean rating per movie for users in this cluster (ignore 0 = unseen)
            avg_ratings = cluster_users.replace(0, np.nan).mean(axis=0)
            top = avg_ratings.nlargest(top_movies).index.tolist()

            rows.append({
                "cluster": cluster_id,
                "size": int(size),
                "top_movies": " | ".join(top),
            })

        return pd.DataFrame(rows)

    def get_user_cluster(self, user_id: int) -> int:
        """Return the cluster ID for a given user."""
        self._assert_fitted()
        if user_id not in self.matrix.index:
            raise ValueError(f"User {user_id} not found.")
        idx = self.matrix.index.get_loc(user_id)
        return int(self.labels[idx])

    def users_in_same_cluster(self, user_id: int) -> list[int]:
        """Return all user IDs in the same cluster as the given user."""
        self._assert_fitted()
        cluster_id = self.get_user_cluster(user_id)
        mask = self.labels == cluster_id
        return self.matrix.index[mask].tolist()

    # ─────────────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────────────

    def _assert_fitted(self):
        if self.matrix is None or self.labels is None:
            raise RuntimeError("Clusterer not fitted. Call clusterer.fit(matrix) first.")
