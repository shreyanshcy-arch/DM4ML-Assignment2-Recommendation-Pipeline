from pathlib import Path
import json
import pandas as pd

def validate(users, movies, ratings, report_path="reports/validation_report.json",
             errors_path="data/errors/invalid_ratings.csv"):
    """Validate required fields, duplicates, types, rating range and foreign keys."""
    required = {"userId", "movieId", "rating", "timestamp"}
    missing_cols = sorted(required - set(ratings.columns))
    if missing_cols:
        raise ValueError(f"Ratings file missing required columns: {missing_cols}")

    r = ratings.copy()
    r["userId"] = pd.to_numeric(r["userId"], errors="coerce")
    r["movieId"] = pd.to_numeric(r["movieId"], errors="coerce")
    r["rating"] = pd.to_numeric(r["rating"], errors="coerce")
    r["timestamp"] = pd.to_numeric(r["timestamp"], errors="coerce")

    reasons = pd.Series("", index=r.index, dtype="object")
    def mark(mask, reason):
        reasons.loc[mask] = reasons.loc[mask].apply(
            lambda x: f"{x}; {reason}".strip("; ") if x else reason
        )
    mark(r[list(required)].isna().any(axis=1), "missing_or_invalid_required_value")
    mark(~r["rating"].between(0.5, 5.0), "rating_out_of_range")
    mark(r.duplicated(["userId", "movieId"], keep="first"), "duplicate_user_movie")
    mark(~r["userId"].isin(users["userId"]), "unknown_user")
    mark(~r["movieId"].isin(movies["movieId"]), "unknown_movie")
    invalid = r[reasons.ne("")].copy()
    invalid["validation_error"] = reasons[reasons.ne("")]
    valid = r[reasons.eq("")].copy()
    valid["userId"] = valid["userId"].astype(int)
    valid["movieId"] = valid["movieId"].astype(int)
    valid["timestamp"] = valid["timestamp"].astype(int)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(errors_path).parent.mkdir(parents=True, exist_ok=True)
    invalid.to_csv(errors_path, index=False)
    report = {
        "status": "PASS" if len(valid) else "FAIL",
        "input_rows": int(len(r)),
        "valid_rows": int(len(valid)),
        "invalid_rows": int(len(invalid)),
        "duplicate_user_movie_rows": int(r.duplicated(["userId", "movieId"]).sum()),
        "missing_or_invalid_required_values": int(r[list(required)].isna().any(axis=1).sum()),
        "rating_range": [0.5, 5.0],
        "valid_user_count": int(valid["userId"].nunique()),
        "valid_movie_count": int(valid["movieId"].nunique())
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if valid.empty:
        raise ValueError("No valid rating rows remain after validation.")
    return valid, report
