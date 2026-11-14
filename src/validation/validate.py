
from pathlib import Path
import json
import pandas as pd


def _validate_id_table(df, id_column, table_name):
    """Validate a user or movie ID table."""
    if id_column not in df.columns:
        raise ValueError(
            f"{table_name} file missing required column: {id_column}"
        )

    result = df.copy()
    numeric_ids = pd.to_numeric(result[id_column], errors="coerce")

    reasons = pd.Series("", index=result.index, dtype="object")

    def mark(mask, reason):
        reasons.loc[mask] = reasons.loc[mask].apply(
            lambda old: f"{old}; {reason}" if old else reason
        )

    mark(numeric_ids.isna(), "missing_or_invalid_id")
    mark(
        numeric_ids.notna() & (numeric_ids % 1 != 0),
        "non_integer_id"
    )
    mark(
        numeric_ids.notna() & (numeric_ids <= 0),
        "non_positive_id"
    )
    mark(
        numeric_ids.duplicated(keep="first") & numeric_ids.notna(),
        "duplicate_id"
    )

    invalid = result[reasons.ne("")].copy()
    invalid["validation_error"] = reasons[reasons.ne("")]

    valid = result[reasons.eq("")].copy()
    valid[id_column] = numeric_ids[reasons.eq("")].astype(int)

    return valid, invalid


def validate(
    users,
    movies,
    ratings,
    report_path="reports/validation_report.json",
    errors_path="data/errors/invalid_ratings.csv"
):
    """
    Validate users, movies and ratings.

    Checks:
    - Required columns
    - Missing and invalid IDs
    - Duplicate IDs and interactions
    - Numeric data types
    - Rating range
    - Timestamp validity
    - User/movie referential integrity

    Returns:
        valid_ratings, report
    """

    # 1. Validate required ID tables
    valid_users, invalid_users = _validate_id_table(
        users, "userId", "Users"
    )

    valid_movies, invalid_movies = _validate_id_table(
        movies, "movieId", "Movies"
    )

    # 2. Validate ratings schema
    required = {"userId", "movieId", "rating", "timestamp"}
    missing_columns = sorted(required - set(ratings.columns))

    if missing_columns:
        raise ValueError(
            f"Ratings file missing required columns: {missing_columns}"
        )

    r = ratings.copy()

    # 3. Convert numeric columns safely
    numeric_columns = ["userId", "movieId", "rating", "timestamp"]

    for column in numeric_columns:
        r[column] = pd.to_numeric(r[column], errors="coerce")

    reasons = pd.Series("", index=r.index, dtype="object")

    def mark(mask, reason):
        reasons.loc[mask] = reasons.loc[mask].apply(
            lambda old: f"{old}; {reason}" if old else reason
        )

    # 4. Missing or invalid required values
    mark(
        r[numeric_columns].isna().any(axis=1),
        "missing_or_invalid_required_value"
    )

    # 5. IDs must be positive integers
    for column in ["userId", "movieId"]:
        mark(
            r[column].notna() & (r[column] % 1 != 0),
            f"{column}_not_integer"
        )
        mark(
            r[column].notna() & (r[column] <= 0),
            f"{column}_not_positive"
        )

    # 6. Rating range validation
    mark(
        ~r["rating"].between(0.5, 5.0),
        "rating_out_of_range"
    )

    # 7. Timestamp validation
    mark(
        r["timestamp"].notna() & (r["timestamp"] <= 0),
        "invalid_timestamp"
    )
    mark(
        r["timestamp"].notna() & (r["timestamp"] % 1 != 0),
        "timestamp_not_integer"
    )

    # 8. Duplicate user/movie interactions
    mark(
        r.duplicated(["userId", "movieId"], keep="first"),
        "duplicate_user_movie"
    )

    # 9. Referential integrity
    valid_user_ids = set(valid_users["userId"])
    valid_movie_ids = set(valid_movies["movieId"])

    mark(
        ~r["userId"].isin(valid_user_ids),
        "unknown_user"
    )
    mark(
        ~r["movieId"].isin(valid_movie_ids),
        "unknown_movie"
    )

    # 10. Separate valid and invalid ratings
    invalid_ratings = r[reasons.ne("")].copy()
    invalid_ratings["validation_error"] = reasons[reasons.ne("")]

    valid_ratings = r[reasons.eq("")].copy()

    for column in ["userId", "movieId", "timestamp"]:
        valid_ratings[column] = valid_ratings[column].astype(int)

    # 11. Create output directories
    report_file = Path(report_path)
    ratings_error_file = Path(errors_path)

    report_file.parent.mkdir(parents=True, exist_ok=True)
    ratings_error_file.parent.mkdir(parents=True, exist_ok=True)

    # 12. Save invalid records separately
    invalid_ratings.to_csv(ratings_error_file, index=False)

    invalid_users.to_csv(
        ratings_error_file.parent / "invalid_users.csv",
        index=False
    )

    invalid_movies.to_csv(
        ratings_error_file.parent / "invalid_movies.csv",
        index=False
    )

    # 13. Build detailed validation report
    report = {
        "status": (
            "PASS"
            if len(valid_ratings) > 0
            and len(invalid_users) == 0
            and len(invalid_movies) == 0
            else "FAIL"
        ),
        "users": {
            "input_rows": int(len(users)),
            "valid_rows": int(len(valid_users)),
            "invalid_rows": int(len(invalid_users)),
            "duplicate_ids": int(
                users["userId"].duplicated().sum()
            ) if "userId" in users.columns else None
        },
        "movies": {
            "input_rows": int(len(movies)),
            "valid_rows": int(len(valid_movies)),
            "invalid_rows": int(len(invalid_movies)),
            "duplicate_ids": int(
                movies["movieId"].duplicated().sum()
            ) if "movieId" in movies.columns else None
        },
        "ratings": {
            "input_rows": int(len(r)),
            "valid_rows": int(len(valid_ratings)),
            "invalid_rows": int(len(invalid_ratings)),
            "duplicate_user_movie_rows": int(
                r.duplicated(["userId", "movieId"]).sum()
            ),
            "missing_or_invalid_required_values": int(
                r[numeric_columns].isna().any(axis=1).sum()
            ),
            "rating_range": [0.5, 5.0],
            "valid_user_count": int(
                valid_ratings["userId"].nunique()
            ),
            "valid_movie_count": int(
                valid_ratings["movieId"].nunique()
            )
        },
        "referential_integrity": {
            "unknown_users": int(
                (~r["userId"].isin(valid_user_ids)).sum()
            ),
            "unknown_movies": int(
                (~r["movieId"].isin(valid_movie_ids)).sum()
            )
        },
        "error_files": {
            "invalid_users": str(
                ratings_error_file.parent / "invalid_users.csv"
            ),
            "invalid_movies": str(
                ratings_error_file.parent / "invalid_movies.csv"
            ),
            "invalid_ratings": str(ratings_error_file)
        }
    }

    with open(report_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    if valid_ratings.empty:
        raise ValueError(
            "No valid rating rows remain after validation."
        )

    if report["status"] == "FAIL":
        raise ValueError(
            "User or movie master-table validation failed. "
            "Check the invalid_users.csv and invalid_movies.csv files."
        )

    return valid_ratings, report
