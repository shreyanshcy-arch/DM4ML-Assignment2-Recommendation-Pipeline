
from pathlib import Path
import math

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error


# --------------------------------------------------
# 1. Discounted Cumulative Gain
# --------------------------------------------------

def _dcg(recommended, relevant, k):

    return sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )


# --------------------------------------------------
# 2. Ideal Discounted Cumulative Gain
# --------------------------------------------------

def _idcg(relevant_count, k):

    ideal_count = min(relevant_count, k)

    return sum(
        1.0 / math.log2(i + 2)
        for i in range(ideal_count)
    )


# --------------------------------------------------
# 3. Parse movie genres
# --------------------------------------------------

def _parse_genres(genres):

    if pd.isna(genres):
        return set()

    if genres == "(no genres listed)":
        return set()

    return set(
        genre.strip()
        for genre in str(genres).split("|")
        if genre.strip()
    )


# --------------------------------------------------
# 4. Genre-based Intra-List Diversity
# --------------------------------------------------

def _genre_diversity(recommended, genre_map):

    if len(recommended) < 2:
        return 0.0

    distances = []

    for i in range(len(recommended)):

        for j in range(i + 1, len(recommended)):

            genres_a = genre_map.get(
                recommended[i], set()
            )

            genres_b = genre_map.get(
                recommended[j], set()
            )

            union = genres_a | genres_b
            intersection = genres_a & genres_b

            if not union:
                distance = 0.0
            else:
                similarity = (
                    len(intersection) / len(union)
                )

                distance = 1.0 - similarity

            distances.append(distance)

    return float(np.mean(distances)) if distances else 0.0


# --------------------------------------------------
# 5. Evaluate one recommendation model
# --------------------------------------------------

def evaluate_model(
    model,
    train,
    test,
    all_items,
    genre_map,
    k=5
):

    # Previously interacted movies by each user
    train_seen = (
        train.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    # Positive interactions in test data
    # Rating >= 4 is treated as relevant
    test_relevant = (
        test[test["rating"] >= 4.0]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    users = sorted(test["userId"].unique())

    precisions = []
    recalls = []
    ndcgs = []

    recommended_union = set()
    diversity_scores = []

    # ----------------------------------------------
    # Generate recommendations for each test user
    # ----------------------------------------------

    for user in users:

        seen = train_seen.get(user, set())

        relevant = test_relevant.get(user, set())

        # Both models exclude previously rated movies
        recs = model.recommend(
            user,
            seen_items=seen,
            k=k
        )

        recs = recs[:k]

        recommended_union.update(recs)

        # ------------------------------------------
        # Precision@K
        # ------------------------------------------

        hits = len(set(recs) & relevant)

        precision = hits / k

        precisions.append(precision)

        # ------------------------------------------
        # Recall@K
        # ------------------------------------------

        if relevant:

            recall = hits / len(relevant)

        else:

            recall = 0.0

        recalls.append(recall)

        # ------------------------------------------
        # NDCG@K
        # ------------------------------------------

        dcg = _dcg(recs, relevant, k)

        idcg = _idcg(len(relevant), k)

        ndcg = dcg / idcg if idcg > 0 else 0.0

        ndcgs.append(ndcg)

        # ------------------------------------------
        # Genre-based Diversity
        # ------------------------------------------

        diversity = _genre_diversity(
            recs,
            genre_map
        )

        if len(recs) >= 2:

            diversity_scores.append(diversity)

    # --------------------------------------------------
    # 6. Catalog Coverage
    # --------------------------------------------------

    coverage = (
        len(recommended_union) / max(1, len(all_items))
    )

    # --------------------------------------------------
    # 7. Average Genre-based Diversity
    # --------------------------------------------------

    diversity = (
        float(np.mean(diversity_scores))
        if diversity_scores
        else 0.0
    )

    # --------------------------------------------------
    # 8. RMSE and MAE Reference Metrics
    # --------------------------------------------------
    # These are item-mean reference predictions.
    # They are NOT model-specific rating predictions.

    item_means = (
        train.groupby("movieId")["rating"]
        .mean()
        .to_dict()
    )

    global_mean = float(train["rating"].mean())

    actual = []
    predicted = []

    for row in test.itertuples():

        actual.append(float(row.rating))

        predicted.append(
            float(
                item_means.get(
                    row.movieId,
                    global_mean
                )
            )
        )

    rmse = float(
        np.sqrt(
            mean_squared_error(actual, predicted)
        )
    )

    mae = float(
        mean_absolute_error(actual, predicted)
    )

    # --------------------------------------------------
    # 9. Return Evaluation Metrics
    # --------------------------------------------------

    return {

        f"Precision@{k}": float(
            np.mean(precisions)
        ),

        f"Recall@{k}": float(
            np.mean(recalls)
        ),

        f"NDCG@{k}": float(
            np.mean(ndcgs)
        ),

        "Coverage": float(coverage),

        "Diversity": diversity,

        "RMSE": rmse,

        "MAE": mae
    }


# --------------------------------------------------
# 10. Compare Both Recommendation Models
# --------------------------------------------------

def compare_models(
    popularity,
    item_cf,
    train,
    test,
    movies,
    output="reports/model_comparison.csv"
):

    # Complete movie catalog
    all_items = set(
        movies["movieId"].unique()
    )

    # Build movie-to-genre mapping
    genre_map = dict(
        zip(
            movies["movieId"],
            movies["genres"].apply(_parse_genres)
        )
    )

    rows = []

    models = [
        ("Popularity", popularity),
        ("Item-item CF", item_cf)
    ]

    # Evaluate each model
    for name, model in models:

        metrics = evaluate_model(
            model=model,
            train=train,
            test=test,
            all_items=all_items,
            genre_map=genre_map,
            k=5
        )

        rows.append({
            "Model": name,
            **metrics
        })

    # Create comparison DataFrame
    result = pd.DataFrame(rows)

    # Ensure output directory exists
    Path(output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save comparison report
    result.to_csv(
        output,
        index=False
    )

    return result
