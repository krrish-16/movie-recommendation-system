"""
app.py
------
Flask web application providing a browser-based UI for the recommendation system.
Also exposes a simple REST API so all features can be accessed programmatically.

Routes
------
  GET  /                          Home / landing page
  GET  /recommend?user_id=N&n=10  Recommendations for user N
  GET  /cluster_summary           Cluster overview table
  GET  /user_cluster?user_id=N    Which cluster a user belongs to
  GET  /association_rules         Top association rules
  GET  /rules_for_movie?title=X   Rules where movie X is the antecedent
  GET  /watched?user_id=N         Movies already rated by user N
  GET  /api/users                 JSON list of all user IDs

VIVA EXPLANATION:
  Flask is a lightweight WSGI web framework for Python. Here it serves
  as the glue layer: it boots all four modules (preprocess → model →
  clustering → apriori), keeps them in memory, and routes HTTP requests
  to the correct function. The Jinja2 templates render HTML tables so
  non-technical stakeholders can interact with the system via a browser.
"""

import random
from flask import Flask, render_template_string, request, jsonify

# ── Project modules ─────────────────────────────────────────────────────────
from preprocess import load_data, preprocess, build_user_movie_matrix
from model import CollaborativeFilteringModel
from clustering import UserClusterer
from apriori import AprioriMiner

app = Flask(__name__)

# ── Global state (loaded once at startup) ────────────────────────────────────
print("\n" + "═" * 60)
print("  🎬  Movie Recommendation System — Starting Up")
print("═" * 60 + "\n")

movies_df, ratings_df = load_data()
merged_df, filtered_ratings = preprocess(movies_df, ratings_df)
matrix = build_user_movie_matrix(merged_df)

cf_model = CollaborativeFilteringModel(n_neighbours=20)
cf_model.fit(matrix)

clusterer = UserClusterer(n_clusters=5)
clusterer.fit(matrix)
cluster_df = clusterer.cluster_summary(top_movies=5)

miner = AprioriMiner(min_support=0.03, min_confidence=0.10, min_lift=1.0)
rules_df = miner.mine(matrix)

ALL_USER_IDS = sorted(matrix.index.tolist())
SAMPLE_USER  = ALL_USER_IDS[0]

print("\n" + "═" * 60)
print("  ✅  All modules loaded. Visit http://127.0.0.1:5000")
print("═" * 60 + "\n")

# ── HTML Template ─────────────────────────────────────────────────────────────
BASE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🎬 Movie Recommender</title>
<style>
  :root{--bg:#0f0f13;--card:#1a1a24;--accent:#e50914;--accent2:#f5c518;
        --text:#e8e8f0;--muted:#8888aa;--border:#2a2a3a;--radius:12px}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
       min-height:100vh}
  header{background:linear-gradient(135deg,#1a0a0a,#0f0f1f);border-bottom:2px solid var(--accent);
         padding:20px 40px;display:flex;align-items:center;gap:16px}
  header h1{font-size:1.8rem;color:#fff}
  header span{font-size:2rem}
  nav{padding:12px 40px;background:var(--card);border-bottom:1px solid var(--border);
      display:flex;gap:12px;flex-wrap:wrap}
  nav a{color:var(--accent2);text-decoration:none;padding:6px 14px;border-radius:20px;
        border:1px solid var(--border);font-size:.85rem;transition:.2s}
  nav a:hover{background:var(--accent);color:#fff;border-color:var(--accent)}
  .content{padding:32px 40px;max-width:1200px;margin:0 auto}
  .card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
        padding:24px;margin-bottom:24px}
  .card h2{margin-bottom:16px;color:var(--accent2);font-size:1.2rem}
  form{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}
  input,select{background:#0f0f1f;border:1px solid var(--border);color:var(--text);
               padding:8px 14px;border-radius:8px;font-size:.9rem}
  button{background:var(--accent);color:#fff;border:none;padding:8px 20px;
         border-radius:8px;cursor:pointer;font-size:.9rem;transition:.2s}
  button:hover{opacity:.85}
  table{width:100%;border-collapse:collapse;font-size:.88rem}
  th{background:#12121c;padding:10px 14px;text-align:left;color:var(--accent2);
     border-bottom:1px solid var(--border)}
  td{padding:9px 14px;border-bottom:1px solid #1e1e2e;vertical-align:top}
  tr:hover td{background:#1e1e2e}
  .badge{display:inline-block;padding:2px 8px;border-radius:10px;
         background:#1e1e3a;font-size:.75rem;margin:1px}
  .stat{font-size:2rem;font-weight:700;color:var(--accent2)}
  .stat-label{font-size:.8rem;color:var(--muted);margin-top:4px}
  .stats-row{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:20px}
  .stat-box{background:#12121c;border-radius:8px;padding:16px 24px;text-align:center}
  .empty{color:var(--muted);font-style:italic;padding:16px 0}
</style>
</head>
<body>
<header><span>🎬</span><h1>Movie Recommendation System</h1></header>
<nav>
  <a href="/">🏠 Home</a>
  <a href="/recommend?user_id={{ sample_user }}&n=10">⭐ Recommendations</a>
  <a href="/cluster_summary">👥 Clusters</a>
  <a href="/association_rules">🔗 Association Rules</a>
  <a href="/watched?user_id={{ sample_user }}">👁 Watched</a>
</nav>
<div class="content">{% block body %}{% endblock %}</div>
</body></html>
"""

HOME_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block body %}{% endblock %}", """
<div class="stats-row">
  <div class="stat-box"><div class="stat">{{ n_users }}</div><div class="stat-label">Users</div></div>
  <div class="stat-box"><div class="stat">{{ n_movies }}</div><div class="stat-label">Movies</div></div>
  <div class="stat-box"><div class="stat">{{ n_ratings }}</div><div class="stat-label">Ratings</div></div>
  <div class="stat-box"><div class="stat">{{ n_rules }}</div><div class="stat-label">Assoc. Rules</div></div>
  <div class="stat-box"><div class="stat">5</div><div class="stat-label">Clusters</div></div>
</div>
<div class="card">
  <h2>⭐ Get Movie Recommendations</h2>
  <form action="/recommend" method="get">
    <input name="user_id" placeholder="User ID" value="{{ sample_user }}" style="width:120px">
    <input name="n" placeholder="Top N" value="10" style="width:80px">
    <button type="submit">Recommend</button>
  </form>
</div>
<div class="card">
  <h2>👁 See Watched Movies</h2>
  <form action="/watched" method="get">
    <input name="user_id" placeholder="User ID" value="{{ sample_user }}" style="width:120px">
    <button type="submit">Show Watched</button>
  </form>
</div>
<div class="card">
  <h2>🔗 Rules for a Movie</h2>
  <form action="/rules_for_movie" method="get">
    <input name="title" placeholder="Movie title (partial)" style="width:280px">
    <button type="submit">Find Rules</button>
  </form>
</div>
""")

TABLE_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block body %}{% endblock %}", """
<div class="card">
  <h2>{{ title }}</h2>
  {% if subtitle %}<p style="color:var(--muted);margin-bottom:16px">{{ subtitle }}</p>{% endif %}
  {% if rows %}
  <table>
    <thead><tr>{% for h in headers %}<th>{{ h }}</th>{% endfor %}</tr></thead>
    <tbody>
      {% for row in rows %}
      <tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}<p class="empty">No results found.</p>{% endif %}
</div>
""")


# ── Helper ────────────────────────────────────────────────────────────────────

def _render_table(title, df, subtitle=""):
    if df.empty:
        rows, headers = [], []
    else:
        headers = df.columns.tolist()
        rows = df.values.tolist()
    return render_template_string(
        TABLE_TEMPLATE,
        title=title,
        subtitle=subtitle,
        headers=headers,
        rows=rows,
        sample_user=SAMPLE_USER,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template_string(
        HOME_TEMPLATE,
        n_users=matrix.shape[0],
        n_movies=matrix.shape[1],
        n_ratings=len(filtered_ratings),
        n_rules=len(rules_df),
        sample_user=SAMPLE_USER,
    )


@app.route("/recommend")
def recommend():
    user_id = int(request.args.get("user_id", SAMPLE_USER))
    top_n = int(request.args.get("n", 10))
    try:
        df = cf_model.recommend(user_id, top_n=top_n)
        cluster_id = clusterer.get_user_cluster(user_id)
        subtitle = f"User {user_id} | Cluster {cluster_id} | Top {top_n} unseen movies"
    except Exception as e:
        df = pd.DataFrame({"error": [str(e)]})
        subtitle = ""
    import pandas as pd
    return _render_table(f"⭐ Recommendations for User {user_id}", df, subtitle)


@app.route("/watched")
def watched():
    import pandas as pd
    user_id = int(request.args.get("user_id", SAMPLE_USER))
    try:
        df = cf_model.get_watched(user_id, top_n=20)
    except Exception as e:
        df = pd.DataFrame({"error": [str(e)]})
    return _render_table(f"👁 Movies Watched by User {user_id}", df)


@app.route("/cluster_summary")
def cluster_summary():
    return _render_table("👥 User Cluster Summary", cluster_df,
                         subtitle="K-Means clusters (K=5). Top movies = highest avg rating within cluster.")


@app.route("/user_cluster")
def user_cluster():
    import pandas as pd
    user_id = int(request.args.get("user_id", SAMPLE_USER))
    try:
        cid = clusterer.get_user_cluster(user_id)
        peers = clusterer.users_in_same_cluster(user_id)
        df = pd.DataFrame({
            "user_id": [user_id],
            "cluster": [cid],
            "cluster_size": [len(peers)],
            "sample_peers": [str(random.sample(peers, min(5, len(peers))))],
        })
    except Exception as e:
        df = pd.DataFrame({"error": [str(e)]})
    return _render_table(f"User {user_id} Cluster Info", df)


@app.route("/association_rules")
def association_rules_route():
    if rules_df.empty:
        import pandas as pd
        df = pd.DataFrame({"message": ["No rules generated. Try lowering min_support."]})
    else:
        df = rules_df[["antecedents", "consequents", "support", "confidence", "lift"]].head(30)
    return _render_table("🔗 Association Rules (Top 30 by Lift)", df,
                         subtitle="Users who watched the antecedent movie(s) also watched the consequent.")


@app.route("/rules_for_movie")
def rules_for_movie():
    title = request.args.get("title", "")
    try:
        df = miner.rules_for_movie(title)
    except Exception as e:
        import pandas as pd
        df = pd.DataFrame({"error": [str(e)]})
    return _render_table(f"🔗 Rules for: '{title}'", df)


@app.route("/api/users")
def api_users():
    return jsonify({"user_ids": ALL_USER_IDS[:100], "total": len(ALL_USER_IDS)})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
