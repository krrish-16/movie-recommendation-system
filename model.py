"""
model.py
--------
Collaborative Filtering recommendation engine using Cosine Similarity.

How it works
------------
1. We represent each user as a vector of their movie ratings.
2. We compute the cosine similarity between all pairs of users.
3. For a target user, we find their K most similar neighbours.
4. We aggregate the neighbours' ratings (weighted by similarity) to
   predict scores for movies the target user has NOT yet seen.
5. We return the top-N movies with the highest predicted scores.

VIVA EXPLANATION:
  Collaborative Filtering (CF) is a key technique in recommender systems.
  It assumes: "users with similar past behaviour will have similar future
  preferences." We use User-Based CF here with cosine similarity as the
  distance metric. Cosine similarity measures the angle between two rating
  vectors, making it robust to different rating scales between users.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringModel:
    """
    User-based Collaborative Filtering recommender.

    Parameters
    ----------
    n_neighbours : int
        Number of similar users to consider when predicting ratings.
    """

    def __init__(self, n_neighbours: int = 20):
        self.n_neighbours = n_neighbours
        self.matrix: pd.DataFrame | None = None          # user × movie matrix
        self.similarity_matrix: np.ndarray | None = None # user × user cosine sim

    # ─────────────────────────────────────────────────────────────────────────
    # Fit
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, user_movie_matrix: pd.DataFrame) -> None:
        """
        Compute and store the user-user cosine similarity matrix.

        Parameters
        ----------
        user_movie_matrix : DataFrame
            Rows = users, columns = movie titles, values = ratings (0 = not rated).
        """
        self.matrix = user_movie_matrix

        print("[Model] Computing user-user cosine similarity …")
        # sklearn's cosine_similarity expects shape (n_samples, n_features)
        self.similarity_matrix = cosine_similarity(user_movie_matrix.values)

        print(f"[Model] Similarity matrix shape: {self.similarity_matrix.shape}")
        print("[Model] Model ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # Recommend
    # ─────────────────────────────────────────────────────────────────────────

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Recommend top-N movies for a given user.

        Steps
        -----
        1. Find the row index of the target user.
        2. Extract the top-K most similar neighbours (excluding the user).
        3. Compute a weighted average predicted rating for each unseen movie.
        4. Return the top-N movies sorted by predicted rating.

        Parameters
        ----------
        user_id : int   Existing userId in the matrix.
        top_n   : int   Number of recommendations to return.

        Returns
        -------
        DataFrame with columns [rank, title, predicted_rating]
        """
        self._assert_fitted()

        if user_id not in self.matrix.index:
            raise ValueError(
                f"User {user_id} not found. "
                f"Valid range: {self.matrix.index.min()} – {self.matrix.index.max()}"
            )

        # 1. Locate user in the matrix
        user_idx = self.matrix.index.get_loc(user_id)
        user_ratings = self.matrix.iloc[user_idx].values  # 1-D array

        # 2. Find K nearest neighbours (highest cosine similarity, excluding self)
        sim_scores = self.similarity_matrix[user_idx].copy()
        sim_scores[user_idx] = -1  # exclude the user themselves
        neighbour_indices = np.argsort(sim_scores)[::-1][: self.n_neighbours]
        neighbour_sims = sim_scores[neighbour_indices]

        # 3. Identify movies the target user has NOT yet rated
        unseen_mask = user_ratings == 0
        unseen_indices = np.where(unseen_mask)[0]

        if len(unseen_indices) == 0:
            return pd.DataFrame(columns=["rank", "title", "predicted_rating"])

        # 4. Weighted average of neighbour ratings for unseen movies
        neighbour_ratings = self.matrix.values[neighbour_indices][:, unseen_indices]
        # shape: (n_neighbours, n_unseen_movies)

        # Avoid division by zero for movies none of the neighbours rated
        sim_sum = np.abs(neighbour_sims).sum()
        if sim_sum == 0:
            return pd.DataFrame(columns=["rank", "title", "predicted_rating"])

        predicted = (neighbour_sims @ neighbour_ratings) / sim_sum
        # predicted shape: (n_unseen_movies,)

        # 5. Build result DataFrame
        unseen_titles = self.matrix.columns[unseen_indices]
        results = pd.DataFrame({
            "title": unseen_titles,
            "predicted_rating": predicted,
        })
        results = (
            results[results["predicted_rating"] > 0]
            .sort_values("predicted_rating", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        results.insert(0, "rank", results.index + 1)
        results["predicted_rating"] = results["predicted_rating"].round(3)

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Watched movies
    # ─────────────────────────────────────────────────────────────────────────

    def get_watched(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Return movies already rated by the user (highest rated first).

        Parameters
        ----------
        user_id : int
        top_n   : int  Maximum number of entries to return.

        Returns
        -------
        DataFrame with columns [title, rating]
        """
        self._assert_fitted()

        if user_id not in self.matrix.index:
            raise ValueError(f"User {user_id} not found.")

        row = self.matrix.loc[user_id]
        watched = row[row > 0].sort_values(ascending=False).head(top_n)
        df = watched.reset_index()
        df.columns = ["title", "rating"]
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────────────

    def _assert_fitted(self):
        if self.matrix is None or self.similarity_matrix is None:
            raise RuntimeError("Model is not fitted. Call model.fit(matrix) first.")
