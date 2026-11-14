from pathlib import Path
import zipfile
import urllib.request
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def ingest(data_dir="data/raw"):
    """Download MovieLens latest-small and load users, movies and ratings."""
    raw = Path(data_dir)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "ml-latest-small.zip"
    extract_dir = raw / "ml-latest-small"
    if not extract_dir.exists():
        if not zip_path.exists():
            print("Downloading MovieLens ml-latest-small...")
            urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw)
    ratings = pd.read_csv(extract_dir / "ratings.csv")
    movies = pd.read_csv(extract_dir / "movies.csv")
    # MovieLens small does not provide a separate user profile file.
    users = pd.DataFrame({"userId": sorted(ratings["userId"].dropna().unique())})
    return users, movies, ratings
