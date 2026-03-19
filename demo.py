"""
demo.py
-------
Non-interactive demo script that prints sample output for ALL features.
Perfect for showing outputs during a viva / presentation without user input.

Run with:
    python demo.py
"""

import random
import pandas as pd

# ── Project modules ─────────────────────────────────────────────────────────
from preprocess import load_data, preprocess, build_user_movie_matrix
from model import CollaborativeFilteringModel
from clustering import UserClusterer
from apriori import AprioriMiner

SEP = "═" * 62


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_df(df, max_rows=10):
    if df.empty:
        print("  (no results)")
        return
    print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"  … and {len(df) - max_rows} more rows")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & Preprocess
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 1 · Data Loading & Preprocessing")

movies_df, ratings_df = load_data()

print(f"\n▶ Raw movies sample:")
print_df(movies_df.head(5))

print(f"\n▶ Raw ratings sample:")
print_df(ratings_df.head(5))

merged_df, filtered_ratings = preprocess(movies_df, ratings_df)

print(f"\n▶ Merged & cleaned sample:")
print_df(merged_df[["userId", "movieId", "title", "genres", "rating"]].head(5))

matrix = build_user_movie_matrix(merged_df)

print(f"\n▶ User-Movie matrix (first 5 users × first 5 movies):")
print(matrix.iloc[:5, :5].to_string())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Collaborative Filtering Recommendations
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 2 · Collaborative Filtering Recommendations")

cf = CollaborativeFilteringModel(n_neighbours=20)
cf.fit(matrix)

sample_users = random.sample(sorted(matrix.index.tolist()), min(3, len(matrix)))

for uid in sample_users:
    watched = cf.get_watched(uid, top_n=5)
    recs    = cf.recommend(uid, top_n=8)

    print(f"\n{'─'*50}")
    print(f"  User {uid} — Already watched (top 5):")
    print_df(watched, max_rows=5)

    print(f"\n  User {uid} — Top 8 Recommendations:")
    print_df(recs, max_rows=8)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — K-Means Clustering
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 3 · K-Means User Clustering  (K = 5)")

clusterer = UserClusterer(n_clusters=5)
clusterer.fit(matrix)

summary = clusterer.cluster_summary(top_movies=4)
print("\n▶ Cluster Summary:")
print_df(summary, max_rows=10)

# Show a couple of sample users and their clusters
print("\n▶ Sample User → Cluster assignments:")
for uid in sample_users:
    cid   = clusterer.get_user_cluster(uid)
    peers = clusterer.users_in_same_cluster(uid)
    print(f"  User {uid:>5d}  →  Cluster {cid}  ({len(peers)} users in cluster)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Association Rule Mining (Apriori)
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 4 · Association Rule Mining — Apriori")

miner = AprioriMiner(
    min_rating=3.5,
    min_support=0.03,
    min_confidence=0.10,   # lower confidence threshold for synthetic/sparse data
    min_lift=1.0,          # lower lift threshold for synthetic/sparse data
)
rules = miner.mine(matrix)

print(f"\n▶ Top 15 Rules (sorted by Lift):")
if not rules.empty:
    print_df(rules[["antecedents", "consequents", "support", "confidence", "lift"]], max_rows=15)
else:
    print("  (no rules generated — try lowering min_support in apriori.py)")

# Search rules for a specific movie
if not rules.empty:
    sample_movie = rules["antecedents"].iloc[0]
    found = miner.rules_for_movie(sample_movie)
    print(f"\n▶ Rules where antecedent = '{sample_movie}':")
    print_df(found[["antecedents", "consequents", "support", "confidence", "lift"]], max_rows=10)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

section("SUMMARY")
print(f"\n  ✅ Users              : {matrix.shape[0]}")
print(f"  ✅ Movies             : {matrix.shape[1]}")
print(f"  ✅ Ratings (filtered) : {len(filtered_ratings)}")
print(f"  ✅ K-Means clusters   : 5")
print(f"  ✅ Association rules  : {len(rules)}")
print(f"\n  🚀 Flask Web UI → run:  python app.py")
print(f"  🖥  CLI Demo      → run:  python cli.py\n")
