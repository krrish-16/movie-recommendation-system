"""
generate_data.py
----------------
Downloads the MovieLens small dataset or generates synthetic data as fallback.
Run this FIRST before anything else.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

DATA_DIR = "data"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_movielens():
    """
    Attempt to download the MovieLens small dataset (~1MB).
    Falls back to generating synthetic data if download fails.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    movies_path = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")

    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        print("[INFO] Dataset already exists. Skipping download.")
        return

    print("[INFO] Attempting to download MovieLens dataset...")
    try:
        zip_path = os.path.join(DATA_DIR, "ml-latest-small.zip")
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            # Extract only movies.csv and ratings.csv
            for name in z.namelist():
                if name.endswith("movies.csv"):
                    with z.open(name) as src, open(movies_path, "wb") as dst:
                        dst.write(src.read())
                elif name.endswith("ratings.csv"):
                    with z.open(name) as src, open(ratings_path, "wb") as dst:
                        dst.write(src.read())

        os.remove(zip_path)
        print("[SUCCESS] MovieLens dataset downloaded.")

    except Exception as e:
        print(f"[WARNING] Download failed ({e}). Generating synthetic dataset...")
        _generate_synthetic_data(movies_path, ratings_path)


def _generate_synthetic_data(movies_path: str, ratings_path: str):
    """
    Generate a realistic synthetic MovieLens-style dataset with 200 movies
    and 500 users for offline / no-internet environments.
    """
    np.random.seed(42)

    # ── Movies ──────────────────────────────────────────────────────────────
    titles = [
        "The Shawshank Redemption", "The Godfather", "The Dark Knight",
        "Pulp Fiction", "Schindler's List", "The Lord of the Rings: The Return of the King",
        "The Good, the Bad and the Ugly", "12 Angry Men", "Forrest Gump",
        "Fight Club", "Inception", "The Matrix", "Goodfellas", "Se7en",
        "The Silence of the Lambs", "Interstellar", "City of God", "The Usual Suspects",
        "Leon: The Professional", "Spirited Away", "Saving Private Ryan",
        "The Green Mile", "Parasite", "Life Is Beautiful", "The Prestige",
        "The Lion King", "Back to the Future", "Avengers: Endgame", "Django Unchained",
        "The Departed", "Whiplash", "The Pianist", "Gladiator", "The Wolf of Wall Street",
        "Inglourious Basterds", "Eternal Sunshine of the Spotless Mind",
        "2001: A Space Odyssey", "Apocalypse Now", "Casablanca", "Memento",
        "Psycho", "Rear Window", "Once Upon a Time in the West",
        "The Intouchables", "Toy Story", "Full Metal Jacket", "The Truman Show",
        "A Beautiful Mind", "Cinema Paradiso", "Braveheart",
        "Lawrence of Arabia", "Come and See", "Oldboy", "Requiem for a Dream",
        "Pan's Labyrinth", "Das Boot", "Alien", "Blade Runner", "Aliens",
        "Amadeus", "Sunset Blvd.", "Paths of Glory", "Vertigo", "Stalker",
        "WALL·E", "Raging Bull", "Modern Times", "Singin' in the Rain",
        "Mulholland Drive", "There Will Be Blood", "No Country for Old Men",
        "Children of Men", "The Grand Budapest Hotel", "Mad Max: Fury Road",
        "Moonlight", "La La Land", "Get Out", "Arrival", "Dunkirk",
        "The Shape of Water", "Three Billboards Outside Ebbing, Missouri",
        "Roma", "Cold War", "1917", "Joker", "Once Upon a Time in Hollywood",
        "Portrait of a Lady on Fire", "The Lighthouse", "Midsommar",
        "Knives Out", "Tenet", "Soul", "Nomadland", "The Father",
        "Judas and the Black Messiah", "Sound of Metal", "Minari",
        "Promising Young Woman", "The Trial of the Chicago 7",
        "News of the World", "Pieces of a Woman", "Another Round",
        "Quo Vadis, Aida?", "Collective",
        "Spider-Man: Into the Spider-Verse", "Hereditary",
        "Sorry to Bother You", "First Reformed",
        "A Quiet Place", "If Beale Street Could Talk", "Roma",
        "Burning", "Shoplifters", "Capernaum", "Never Look Away",
        "The Favourite", "At Eternity's Gate", "First Man",
        "BlacKkKlansman", "Bohemian Rhapsody", "Green Book", "Vice",
        "Mary Queen of Scots", "The Wife", "Wildlife",
        "Leave No Trace", "Lean on Pete", "First Reformed",
        "Eighth Grade", "Blindspotting", "Upgrade",
        "Mission: Impossible - Fallout", "Avengers: Infinity War",
        "Black Panther", "The Favourite", "Isle of Dogs",
        "Three Identical Strangers", "Won't You Be My Neighbor?",
        "Free Solo", "RBG", "Minding the Gap",
        "Dark Waters", "Little Women", "Marriage Story",
        "The Irishman", "Ford v Ferrari", "Parasite",
        "Doctor Sleep", "Ad Astra", "Hustlers",
        "Booksmart", "Rocketman", "Yesterday",
        "The Peanut Butter Falcon", "Waves", "A Hidden Life",
        "Judy", "Dolemite Is My Name", "The Two Popes",
        "Knives Out", "Jojo Rabbit", "Weathering With You",
        "Cats", "Richard Jewell", "21 Bridges",
        "Midway", "Harriet", "Terminator: Dark Fate",
        "Maleficent: Mistress of Evil", "Zombieland: Double Tap",
        "Abominable", "The Addams Family", "Downton Abbey",
        "It Chapter Two", "Good Boys", "Fast & Furious Presents: Hobbs & Shaw",
        "Once Upon a Time in Hollywood", "Dora and the Lost City of Gold",
        "The Lion King", "Stuber", "Spider-Man: Far from Home",
        "Men in Black: International", "Toy Story 4", "Anna",
        "John Wick: Chapter 3", "Brightburn", "A Dog's Journey",
        "Pokémon Detective Pikachu", "Breakthrough", "Shazam!",
        "Us", "Captain Marvel", "Alita: Battle Angel",
        "How to Train Your Dragon: The Hidden World", "The Lego Movie 2",
        "Glass", "Replicas", "Escape Room",
        "Aquaman", "Bumblebee", "Mary Poppins Returns",
        "Into the Spider-Verse", "The Mule", "Mortal Engines",
        "Creed II", "Boy Erased", "Beautiful Boy",
        "The Grinch", "Bohemian Rhapsody", "Venom",
        "A Simple Favor", "The Predator", "The Nun",
        "Searching", "Crazy Rich Asians", "BlacKkKlansman",
        "Mission: Impossible - Fallout", "Mamma Mia! Here We Go Again",
        "Ant-Man and the Wasp", "Sorry to Bother You",
        "Hereditary", "Won't You Be My Neighbor?",
        "Avengers: Infinity War", "A Quiet Place",
        "Ready Player One", "Love, Simon",
    ]

    genres_pool = [
        "Action", "Adventure", "Animation", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]

    movie_ids = list(range(1, len(titles) + 1))
    genres = [
        "|".join(np.random.choice(genres_pool, size=np.random.randint(1, 4), replace=False))
        for _ in titles
    ]

    movies_df = pd.DataFrame({
        "movieId": movie_ids,
        "title": titles,
        "genres": genres,
    })
    movies_df.to_csv(movies_path, index=False)
    print(f"[INFO] Generated {len(movies_df)} synthetic movies → {movies_path}")

    # ── Ratings ──────────────────────────────────────────────────────────────
    n_users = 500
    records = []
    for user_id in range(1, n_users + 1):
        n_rated = np.random.randint(10, 60)
        movie_sample = np.random.choice(movie_ids, size=n_rated, replace=False)
        for mid in movie_sample:
            rating = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            records.append((user_id, mid, rating, 964982703))

    ratings_df = pd.DataFrame(records, columns=["userId", "movieId", "rating", "timestamp"])
    ratings_df.to_csv(ratings_path, index=False)
    print(f"[INFO] Generated {len(ratings_df)} synthetic ratings → {ratings_path}")


if __name__ == "__main__":
    download_movielens()
