from pathlib import Path
import pandas as pd

def prepare(ratings, movies, processed_dir="data/processed", test_fraction=0.20):
    """Sort interactions chronologically and make a global time-based holdout."""
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = ratings.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    data = data.sort_values(["timestamp", "userId", "movieId"]).reset_index(drop=True)
    split_at = int(len(data) * (1 - test_fraction))
    train = data.iloc[:split_at].copy()
    test = data.iloc[split_at:].copy()
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    data.to_csv(out / "clean_interactions.csv", index=False)
    return train, test, data
