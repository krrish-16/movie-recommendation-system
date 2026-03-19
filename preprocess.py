"""
preprocess.py
-------------
Handles all data loading and preprocessing steps:
  - Load movies.csv and ratings.csv
  - Merge datasets
  - Handle missing values
  - Build the user-movie rating matrix
  - Provide helper utilities used by other modules

VIVA EXPLANATION:
  Preprocessing is the foundation of any data mining pipeline.
  Here we clean raw MovieLens data, merge it, and convert it into
  a user-item matrix where rows = users, columns = movies, and
  cell values = ratings (0 means "not yet rated").
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load movies.csv and ratings.csv from the data/ directory.

    Returns
    -------
    movies  : DataFrame  [movieId, title, genres]
    ratings : DataFrame  [userId, movieId, rating, timestamp]
    """
    _check_files_exist()

    movies = pd.read_csv(MOVIES_CSV)
    ratings = pd.read_csv(RATINGS_CSV)

    print(f"[Preprocess] Loaded {len(movies):,} movies and {len(ratings):,} ratings.")
    return movies, ratings


def preprocess(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    min_ratings_per_user: int = 5,
    min_ratings_per_movie: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and merge the raw DataFrames.

    Steps
    -----
    1. Drop rows with null values in critical columns.
    2. Remove duplicate (userId, movieId) pairs (keep last rating).
    3. Filter out users/movies with very few interactions (cold-start fix).
    4. Merge ratings with movie titles for a unified DataFrame.

    Returns
    -------
    merged_df      : Cleaned merged DataFrame
    filtered_ratings : Cleaned ratings-only DataFrame
    """
    print("\n[Preprocess] ── Cleaning ──────────────────────────────────────────")

    # 1. Drop rows missing critical values
    before = len(ratings)
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    movies = movies.dropna(subset=["movieId", "title"])
    print(f"  Dropped {before - len(ratings):,} rating rows with null values.")

    # 2. Remove duplicate (user, movie) pairs
    before = len(ratings)
    ratings = ratings.drop_duplicates(subset=["userId", "movieId"], keep="last")
    print(f"  Removed {before - len(ratings):,} duplicate (user, movie) entries.")

    # 3. Filter cold-start users and rarely rated movies
    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()
    active_users = user_counts[user_counts >= min_ratings_per_user].index
    popular_movies = movie_counts[movie_counts >= min_ratings_per_movie].index
    before = len(ratings)
    ratings = ratings[
        ratings["userId"].isin(active_users) & ratings["movieId"].isin(popular_movies)
    ]
    print(
        f"  Kept {len(active_users):,} users (≥{min_ratings_per_user} ratings) and "
        f"{len(popular_movies):,} movies (≥{min_ratings_per_movie} ratings)."
    )
    print(f"  Dropped {before - len(ratings):,} low-activity rows.")

    # 4. Merge with movie metadata
    merged_df = ratings.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    merged_df = merged_df.dropna(subset=["title"])

    print(f"\n[Preprocess] Final dataset: {len(merged_df):,} rows, "
          f"{merged_df['userId'].nunique():,} users, "
          f"{merged_df['movieId'].nunique():,} movies.")

    return merged_df, ratings


def build_user_movie_matrix(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a user × movie pivot table of ratings.

    Rows    = users  (userId)
    Columns = movies (title)
    Values  = rating (0.0 where user hasn't rated the movie)

    Returns
    -------
    matrix : DataFrame  shape (n_users, n_movies), NaN → 0
    """
    matrix = merged_df.pivot_table(
        index="userId",
        columns="title",
        values="rating",
        aggfunc="mean",   # average if a user rated the same movie twice
    )
    matrix = matrix.fillna(0)

    print(f"\n[Preprocess] User-Movie matrix shape: {matrix.shape}  "
          f"({matrix.shape[0]} users × {matrix.shape[1]} movies)")
    print(f"  Sparsity: "
          f"{(matrix == 0).sum().sum() / matrix.size * 100:.1f}% zeros")

    return matrix


def get_user_ids(matrix: pd.DataFrame) -> list[int]:
    """Return sorted list of all user IDs present in the matrix."""
    return sorted(matrix.index.tolist())


def get_movie_titles(matrix: pd.DataFrame) -> list[str]:
    """Return sorted list of all movie titles present in the matrix."""
    return sorted(matrix.columns.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_files_exist():
    """Raise a helpful error if the dataset has not been generated yet."""
    missing = [f for f in [MOVIES_CSV, RATINGS_CSV] if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files: {missing}\n"
            "Run `python generate_data.py` first to download or generate the dataset."
        )
