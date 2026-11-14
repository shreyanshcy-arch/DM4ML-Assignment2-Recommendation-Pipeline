from pathlib import Path
import pandas as pd

def transform(train, test, movies, output_dir="data/processed"):
    """Create reusable interaction matrix and summary tables."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    matrix = train.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean")
    matrix.to_csv(out / "user_item_matrix.csv")
    user_summary = train.groupby("userId").agg(
        user_activity_count=("rating", "count"),
        user_average_rating=("rating", "mean"),
        last_interaction=("timestamp", "max")
    ).reset_index()
    item_summary = train.groupby("movieId").agg(
        item_popularity=("rating", "count"),
        item_average_rating=("rating", "mean")
    ).reset_index()
    user_summary.to_csv(out / "user_activity_summary.csv", index=False)
    item_summary.to_csv(out / "item_popularity_summary.csv", index=False)
    return matrix, user_summary, item_summary
