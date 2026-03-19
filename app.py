"""
app.py  —  CineMatch Flask Application
Serves the cinematic UI and all JSON API endpoints.
"""

from flask import Flask, render_template, jsonify, request
from preprocess import load_data, preprocess, build_user_movie_matrix
from model import CollaborativeFilteringModel
from clustering import UserClusterer
from apriori import AprioriMiner

app = Flask(__name__)

print("\n" + "="*60)
print("  CineMatch — Starting Up")
print("="*60 + "\n")

movies_df, ratings_df  = load_data()
merged_df, filtered_df = preprocess(movies_df, ratings_df)
matrix                 = build_user_movie_matrix(merged_df)

cf = CollaborativeFilteringModel(n_neighbours=20)
cf.fit(matrix)

clusterer = UserClusterer(n_clusters=5)
clusterer.fit(matrix)
cluster_summary = clusterer.cluster_summary(top_movies=5)

miner    = AprioriMiner(min_support=0.03, min_confidence=0.10, min_lift=1.0)
rules_df = miner.mine(matrix)

ALL_USERS = sorted(matrix.index.tolist())

print("\n" + "="*60)
print("  Ready at http://127.0.0.1:5000")
print("="*60 + "\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stats")
def api_stats():
    return jsonify({
        "users":   int(matrix.shape[0]),
        "movies":  int(matrix.shape[1]),
        "ratings": int(len(filtered_df)),
        "rules":   int(len(rules_df)),
    })

@app.route("/api/recommend")
def api_recommend():
    try:
        user_id = int(request.args.get("user_id", ALL_USERS[0]))
        n       = int(request.args.get("n", 10))
        recs    = cf.recommend(user_id, top_n=n)
        return jsonify({
            "user_id": user_id,
            "watched_count": int((matrix.loc[user_id] > 0).sum()),
            "recommendations": [
                {"rank": int(r.rank), "title": r.title, "predicted_rating": float(r.predicted_rating)}
                for r in recs.itertuples()
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/watched")
def api_watched():
    try:
        user_id = int(request.args.get("user_id", ALL_USERS[0]))
        watched = cf.get_watched(user_id, top_n=20)
        return jsonify({
            "user_id": user_id,
            "watched": [
                {"title": r.title, "rating": float(r.rating)}
                for r in watched.itertuples()
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/user_cluster")
def api_user_cluster():
    try:
        user_id = int(request.args.get("user_id", ALL_USERS[0]))
        cluster = clusterer.get_user_cluster(user_id)
        peers   = clusterer.users_in_same_cluster(user_id)
        return jsonify({
            "user_id": user_id,
            "cluster": int(cluster),
            "cluster_size": len(peers),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/clusters")
def api_clusters():
    rows = []
    for _, row in cluster_summary.iterrows():
        rows.append({
            "cluster":    int(row["cluster"]),
            "size":       int(row["size"]),
            "top_movies": str(row["top_movies"]),
        })
    return jsonify({"clusters": rows})

@app.route("/api/rules")
def api_rules():
    if rules_df.empty:
        return jsonify({"rules": []})
    rules = []
    for _, row in rules_df.head(20).iterrows():
        rules.append({
            "antecedents": str(row["antecedents"]),
            "consequents": str(row["consequents"]),
            "support":     round(float(row["support"]), 4),
            "confidence":  round(float(row["confidence"]), 4),
            "lift":        round(float(row["lift"]), 3),
        })
    return jsonify({"rules": rules})

@app.route("/api/users")
def api_users():
    return jsonify({"user_ids": ALL_USERS[:200], "total": len(ALL_USERS)})

if __name__ == "__main__":
    import os
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))