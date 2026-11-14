from pathlib import Path
import pandas as pd

def build_features(train, movies, output_dir="data/features"):
    """Build documented user, item and interaction feature tables."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    users = train.groupby("userId").agg(
        user_average_rating=("rating", "mean"),
        user_activity_count=("rating", "count"),
        user_last_interaction=("timestamp", "max")
    ).reset_index()
    items = train.groupby("movieId").agg(
        item_popularity=("rating", "count"),
        item_average_rating=("rating", "mean"),
        item_rating_std=("rating", "std")
    ).reset_index().merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    interactions = train[["userId", "movieId", "rating", "timestamp"]].copy()
    users.to_csv(out / "user_features.csv", index=False)
    items.to_csv(out / "item_features.csv", index=False)
    interactions.to_csv(out / "interaction_features.csv", index=False)
    catalog = pd.DataFrame([
        ["user_average_rating", "User", "Mean rating given by user", "Training interactions only"],
        ["user_activity_count", "User", "Number of training interactions", "Training interactions only"],
        ["user_last_interaction", "User", "Latest interaction timestamp", "Unix timestamp"],
        ["item_popularity", "Item", "Number of training ratings", "Training interactions only"],
        ["item_average_rating", "Item", "Mean rating received by item", "Training interactions only"],
        ["item_rating_std", "Item", "Standard deviation of item ratings", "May be null for one rating"],
        ["rating", "Interaction", "Explicit user rating", "0.5 to 5.0"],
        ["timestamp", "Interaction", "Interaction time", "Unix timestamp"]
    ], columns=["feature", "entity", "definition", "notes"])
    catalog.to_csv("reports/feature_catalog.csv", index=False)
    return users, items, interactions
