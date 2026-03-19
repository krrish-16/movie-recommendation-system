# 🎬 Movie Recommendation System
### Data Mining Project — University Assignment

A complete, end-to-end Movie Recommendation System built with Python,
implementing Collaborative Filtering, K-Means Clustering, and Association
Rule Mining (Apriori) on the MovieLens dataset.

---

## 📁 Project Structure

```
movie_recommender/
│
├── generate_data.py     Downloads MovieLens dataset (or generates synthetic data)
├── preprocess.py        Data loading, cleaning, and user-movie matrix creation
├── model.py             User-based Collaborative Filtering (cosine similarity)
├── clustering.py        K-Means clustering of users
├── apriori.py           Association Rule Mining with Apriori algorithm
├── app.py               Flask web application (browser UI + REST API)
├── cli.py               Interactive command-line interface
├── demo.py              Non-interactive demo — prints all sample outputs
├── requirements.txt     Python dependencies
└── data/                Auto-created by generate_data.py
    ├── movies.csv
    └── ratings.csv
```

---

## ⚙️ Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download / generate the dataset
```bash
python generate_data.py
```
> If you have internet access, this downloads the real MovieLens-small dataset
> (~100k ratings, 9k movies).  Without internet, it generates a realistic
> synthetic dataset automatically.

---

## 🚀 How to Run

### Option A — Non-interactive Demo (see all outputs at once)
```bash
python demo.py
```
Runs all four steps and prints sample outputs. Great for a viva demo.

### Option B — Interactive CLI
```bash
python cli.py
```
Provides a menu-driven interface. Choose a user, get recommendations,
explore clusters, and browse association rules interactively.

### Option C — Flask Web Application
```bash
python app.py
```
Opens a browser UI at **http://127.0.0.1:5000**

Web routes:
| URL | Description |
|-----|-------------|
| `/` | Dashboard with stats |
| `/recommend?user_id=1&n=10` | Top-N recommendations |
| `/watched?user_id=1` | Movies already seen |
| `/cluster_summary` | K-Means cluster overview |
| `/user_cluster?user_id=1` | Which cluster a user belongs to |
| `/association_rules` | Top association rules by lift |
| `/rules_for_movie?title=Matrix` | Rules involving a specific movie |
| `/api/users` | JSON list of user IDs |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.1.4 | Data loading and manipulation |
| numpy | 1.26.4 | Numerical computations |
| scikit-learn | 1.4.0 | Cosine similarity, K-Means, SVD |
| mlxtend | 0.23.1 | Apriori & association_rules |
| flask | 3.0.0 | Web application server |
| scipy | 1.12.0 | Sparse matrix utilities |

---

## 🧠 Module Explanations (Viva Notes)

### `preprocess.py`
Loads `movies.csv` and `ratings.csv`, drops nulls and duplicates, filters
cold-start users/movies, then pivots the data into a **User × Movie matrix**
where each cell is a rating (0 = not seen).

### `model.py` — Collaborative Filtering
Implements **User-Based CF** using **cosine similarity**.

1. Each user is a vector of movie ratings.
2. Cosine similarity measures the angle between vectors (scale-invariant).
3. For a target user, we find K most similar neighbours.
4. We predict ratings for unseen movies as a **weighted average** of
   neighbour ratings (weight = cosine similarity score).
5. Movies already rated are excluded from recommendations.

Key formula:
```
predicted_rating(u, m) = Σ sim(u, v) * r(v, m)  /  Σ |sim(u, v)|
                          v ∈ Neighbours(u)
```

### `clustering.py` — K-Means
Groups the 500+ users into 5 clusters based on rating behaviour.

- Uses **TruncatedSVD** to reduce the high-dimensional rating matrix
  before clustering (improves speed + quality).
- L2-normalises vectors so cosine distance ≈ Euclidean distance.
- Reports the **top movies** per cluster (most loved by that group).
- Enables group recommendations and narrows the CF neighbour search.

### `apriori.py` — Association Rule Mining
Applies the **Apriori algorithm** to find patterns like:
> *"Users who liked Inception also liked Interstellar"*

Steps:
1. **Binarise** the matrix: 1 if rating ≥ 3.5, else 0.
2. Find **frequent itemsets** (movie pairs appearing together ≥ min_support).
3. Generate **association rules** with metrics:
   - **Support** = P(A ∩ B) — how common the co-occurrence is
   - **Confidence** = P(B | A) — how reliably B follows A
   - **Lift** = confidence / P(B) — improvement over random baseline
4. Filter by confidence and lift thresholds.

### `app.py` — Flask Web Application
Lightweight WSGI web server. Boots all modules at startup, keeps them
in memory, and serves results via HTML tables and a JSON API.

### `cli.py` — Command-Line Interface
Menu-driven interactive demo for terminal environments. Useful for
running the project on a headless server or during a terminal viva.

---

## 📊 Sample Outputs

### Recommendations
```
User 37 — Already watched: A Quiet Place (5.0), Toy Story (5.0) …
Recommendations:
 rank  title               predicted_rating
    1  The Truman Show              1.391
    2  Hereditary                   1.304
    3  Amadeus                      1.293
```

### Clusters
```
cluster  size  top_movies
      0    86  A Simple Favor | Lean on Pete | …
      1    99  Weathering With You | The Grinch | …
      2   133  The Trial of the Chicago 7 | …
```

### Association Rules
```
antecedents           consequents           support  confidence  lift
Sorry to Bother You   Bohemian Rhapsody       0.030       0.217  1.87
Bohemian Rhapsody     Sorry to Bother You     0.030       0.259  1.87
```

---

## 📝 Notes

- The synthetic dataset uses randomly generated ratings, so the association
  rules will be weaker than with the real MovieLens dataset. Download the
  real dataset for stronger rules: https://grouplens.org/datasets/movielens/
- Adjust `min_support`, `min_confidence`, `min_lift` in `apriori.py` to
  get more or fewer rules.
- Adjust `n_clusters` in `clustering.py` and `n_neighbours` in `model.py`
  to tune performance.
