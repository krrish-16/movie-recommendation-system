"""
apriori.py
----------
Discovers association rules between movies using the Apriori algorithm.

Interpretation
--------------
  Rule: {Movie A} → {Movie B}   support=0.05, confidence=0.7, lift=3.2
  Means: 5% of users watched both A and B; 70% of users who watched A
  also watched B; that's 3.2× more likely than random chance.

VIVA EXPLANATION:
  Association Rule Mining (ARM) is a technique from market-basket analysis.
  Applied to movies: each user's set of liked/watched movies is a "basket."
  The Apriori algorithm finds frequent itemsets (movie combinations that
  appear together often) and generates rules like "if a user watches X,
  they are likely to also watch Y."

  Key metrics:
  - Support    = P(A ∩ B)          — how common the pair is overall
  - Confidence = P(B | A)          — how often B follows A
  - Lift       = conf / P(B)       — improvement over random baseline
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


class AprioriMiner:
    """
    Mine association rules between movies from user rating data.

    Parameters
    ----------
    min_rating         : float   Threshold — a movie counts as "watched/liked"
                                 only if the user's rating ≥ this value.
    min_support        : float   Minimum support for frequent itemsets.
    min_confidence     : float   Minimum confidence for rules.
    min_lift           : float   Minimum lift for rules.
    max_rules          : int     Cap on number of rules returned.
    """

    def __init__(
        self,
        min_rating: float = 3.5,
        min_support: float = 0.03,
        min_confidence: float = 0.3,
        min_lift: float = 1.5,
        max_rules: int = 50,
    ):
        self.min_rating = min_rating
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_rules = max_rules

        self.rules: pd.DataFrame | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Mine
    # ─────────────────────────────────────────────────────────────────────────

    def mine(self, user_movie_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full Apriori pipeline on the user-movie rating matrix.

        Steps
        -----
        1. Binarise the matrix: 1 if rating ≥ min_rating, else 0.
        2. Run Apriori to find frequent itemsets.
        3. Generate association rules from the frequent itemsets.
        4. Filter by min_confidence and min_lift; sort by lift.

        Parameters
        ----------
        user_movie_matrix : DataFrame  (users × movies, values = ratings)

        Returns
        -------
        rules DataFrame with columns:
            antecedents, consequents, support, confidence, lift
        """
        print(f"\n[Apriori] Binarising matrix (rating ≥ {self.min_rating}) …")
        binary_matrix = (user_movie_matrix >= self.min_rating).astype(bool)

        # Keep only movies rated by at least a minimum fraction of users
        # (already handled by min_support in apriori, but this speeds things up)
        col_support = binary_matrix.mean(axis=0)
        useful_cols = col_support[col_support >= self.min_support * 0.5].index
        binary_matrix = binary_matrix[useful_cols]

        print(f"[Apriori] Matrix shape after filtering: {binary_matrix.shape}")
        print(f"[Apriori] Mining frequent itemsets (min_support={self.min_support}) …")

        try:
            frequent_itemsets = apriori(
                binary_matrix,
                min_support=self.min_support,
                use_colnames=True,
                max_len=2,      # pairs only (A → B) to keep it tractable
            )
        except Exception as e:
            print(f"[Apriori] WARNING: {e}")
            self.rules = pd.DataFrame()
            return self.rules

        if frequent_itemsets.empty:
            print("[Apriori] No frequent itemsets found. "
                  "Try lowering min_support.")
            self.rules = pd.DataFrame()
            return self.rules

        print(f"[Apriori] Found {len(frequent_itemsets)} frequent itemsets.")
        print("[Apriori] Generating association rules …")

        raw_rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence,
        )

        # Filter by lift
        raw_rules = raw_rules[raw_rules["lift"] >= self.min_lift]

        # Sort by lift descending and cap
        self.rules = (
            raw_rules.sort_values("lift", ascending=False)
            .head(self.max_rules)
            .reset_index(drop=True)
        )

        # Convert frozensets to readable strings
        self.rules["antecedents"] = self.rules["antecedents"].apply(
            lambda x: ", ".join(sorted(x))
        )
        self.rules["consequents"] = self.rules["consequents"].apply(
            lambda x: ", ".join(sorted(x))
        )

        # Round for display
        for col in ["support", "confidence", "lift"]:
            self.rules[col] = self.rules[col].round(4)

        print(f"[Apriori] {len(self.rules)} rules after filtering "
              f"(confidence≥{self.min_confidence}, lift≥{self.min_lift}).")
        return self.rules

    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────

    def rules_for_movie(self, movie_title: str) -> pd.DataFrame:
        """
        Return all rules where the given movie appears as an antecedent.

        Parameters
        ----------
        movie_title : str   (partial match is supported)

        Returns
        -------
        Filtered rules DataFrame.
        """
        self._assert_mined()
        mask = self.rules["antecedents"].str.contains(movie_title, case=False, na=False)
        return self.rules[mask].reset_index(drop=True)

    def top_rules(self, n: int = 10) -> pd.DataFrame:
        """Return the top-N rules by lift."""
        self._assert_mined()
        return self.rules.head(n)

    # ─────────────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────────────

    def _assert_mined(self):
        if self.rules is None:
            raise RuntimeError("No rules found. Call miner.mine(matrix) first.")
