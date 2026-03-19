"""
cli.py
------
Interactive Command-Line Interface for the Movie Recommendation System.
Provides a menu-driven demo so you can explore all features without a browser.

Run with:
    python cli.py
"""

import sys
import random

# ── Project modules ─────────────────────────────────────────────────────────
from preprocess import load_data, preprocess, build_user_movie_matrix
from model import CollaborativeFilteringModel
from clustering import UserClusterer
from apriori import AprioriMiner


# ── Colour helpers (works on most terminals) ─────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"

def title(text):     print(f"\n{C.BOLD}{C.CYAN}{'═'*60}{C.RESET}")
def header(text):    print(f"{C.BOLD}{C.YELLOW}  {text}{C.RESET}")
def info(text):      print(f"  {C.GREEN}→{C.RESET}  {text}")
def warn(text):      print(f"  {C.RED}⚠{C.RESET}  {text}")
def sep():           print(f"{C.CYAN}{'─'*60}{C.RESET}")


def print_table(df, max_rows=20):
    """Pretty-print a DataFrame as an ASCII table."""
    if df.empty:
        warn("No results.")
        return
    subset = df.head(max_rows)
    col_widths = {col: max(len(str(col)), subset[col].astype(str).map(len).max())
                  for col in subset.columns}
    header_row = "  ".join(str(col).ljust(col_widths[col]) for col in subset.columns)
    print(f"\n{C.BOLD}  {header_row}{C.RESET}")
    print("  " + "-" * (sum(col_widths.values()) + 2 * len(col_widths)))
    for _, row in subset.iterrows():
        line = "  ".join(str(row[col]).ljust(col_widths[col]) for col in subset.columns)
        print(f"  {line}")
    if len(df) > max_rows:
        print(f"  … and {len(df) - max_rows} more rows")
    print()


# ── Boot ──────────────────────────────────────────────────────────────────────

def boot_system():
    print(f"\n{C.BOLD}{C.MAGENTA}{'═'*60}")
    print("   🎬  Movie Recommendation System — CLI Demo")
    print(f"{'═'*60}{C.RESET}\n")

    print(f"{C.CYAN}[1/4] Loading & preprocessing data …{C.RESET}")
    movies_df, ratings_df = load_data()
    merged_df, filtered_ratings = preprocess(movies_df, ratings_df)
    matrix = build_user_movie_matrix(merged_df)

    print(f"\n{C.CYAN}[2/4] Training collaborative filtering model …{C.RESET}")
    cf = CollaborativeFilteringModel(n_neighbours=20)
    cf.fit(matrix)

    print(f"\n{C.CYAN}[3/4] Clustering users …{C.RESET}")
    clusterer = UserClusterer(n_clusters=5)
    clusterer.fit(matrix)

    print(f"\n{C.CYAN}[4/4] Mining association rules …{C.RESET}")
    miner = AprioriMiner(min_support=0.03, min_confidence=0.3, min_lift=1.5)
    miner.mine(matrix)

    print(f"\n{C.GREEN}✅  All systems ready.{C.RESET}")
    return matrix, cf, clusterer, miner


# ── Menus ─────────────────────────────────────────────────────────────────────

def menu_recommend(matrix, cf):
    title("Collaborative Filtering Recommendations")
    header("User-Based Collaborative Filtering")
    sep()
    user_ids = sorted(matrix.index.tolist())
    sample = random.choice(user_ids)
    info(f"Available user IDs: {user_ids[:10]} … (total: {len(user_ids)})")
    try:
        uid = int(input(f"\n  Enter user ID [{sample}]: ").strip() or sample)
        n   = int(input("  Top N recommendations [10]: ").strip() or 10)
    except ValueError:
        warn("Invalid input."); return

    watched = cf.get_watched(uid, top_n=8)
    print(f"\n  {C.BOLD}Movies already watched by User {uid}:{C.RESET}")
    print_table(watched, max_rows=8)

    recs = cf.recommend(uid, top_n=n)
    print(f"  {C.BOLD}Top {n} Recommendations for User {uid}:{C.RESET}")
    print_table(recs, max_rows=n)


def menu_clusters(matrix, clusterer):
    title("K-Means User Clustering")
    header("Cluster Summary")
    sep()
    summary = clusterer.cluster_summary(top_movies=5)
    print_table(summary, max_rows=10)

    user_ids = sorted(matrix.index.tolist())
    sample = random.choice(user_ids)
    try:
        uid = int(input(f"  Enter user ID to see their cluster [{sample}]: ").strip() or sample)
    except ValueError:
        warn("Invalid input."); return

    cid   = clusterer.get_user_cluster(uid)
    peers = clusterer.users_in_same_cluster(uid)
    info(f"User {uid} → Cluster {cid}  (size: {len(peers)} users)")
    info(f"Sample peers: {random.sample(peers, min(5, len(peers)))}")


def menu_apriori(miner):
    title("Association Rule Mining (Apriori)")
    header("Top Rules by Lift")
    sep()
    top = miner.top_rules(15)
    print_table(top, max_rows=15)

    movie = input("  Search rules for a specific movie (partial title, or ENTER to skip): ").strip()
    if movie:
        rules = miner.rules_for_movie(movie)
        print(f"\n  {C.BOLD}Rules where antecedent contains '{movie}':{C.RESET}")
        print_table(rules, max_rows=15)


def main_menu(matrix, cf, clusterer, miner):
    while True:
        print(f"\n{C.BOLD}{C.CYAN}{'═'*60}")
        print("   MAIN MENU")
        print(f"{'═'*60}{C.RESET}")
        print("  1. 🎯  Get movie recommendations for a user")
        print("  2. 👥  View user clusters (K-Means)")
        print("  3. 🔗  Browse association rules (Apriori)")
        print("  4. 📊  Dataset statistics")
        print("  0. 🚪  Exit")
        print()

        choice = input("  Choose an option: ").strip()

        if choice == "1":
            menu_recommend(matrix, cf)
        elif choice == "2":
            menu_clusters(matrix, clusterer)
        elif choice == "3":
            menu_apriori(miner)
        elif choice == "4":
            title("Dataset Statistics")
            info(f"Total users  : {matrix.shape[0]}")
            info(f"Total movies : {matrix.shape[1]}")
            sparsity = (matrix.values == 0).sum() / matrix.size * 100
            info(f"Matrix sparsity: {sparsity:.1f}%")
        elif choice == "0":
            print(f"\n  {C.GREEN}Goodbye! 🎬{C.RESET}\n")
            sys.exit(0)
        else:
            warn("Invalid choice. Please enter 0–4.")


if __name__ == "__main__":
    matrix, cf, clusterer, miner = boot_system()
    main_menu(matrix, cf, clusterer, miner)
