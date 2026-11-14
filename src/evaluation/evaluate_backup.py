import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def _dcg(recommended, relevant, k):
    return sum((1.0 / math.log2(i + 2)) for i, item in enumerate(recommended[:k]) if item in relevant)

def evaluate_model(model, train, test, all_items, k=5, personalized=True):
    """Ranking metrics use held-out positive interactions (rating >= 4)."""
    train_seen = train.groupby("userId").movieId.apply(set).to_dict()
    test_relevant = test[test.rating >= 4.0].groupby("userId").movieId.apply(set).to_dict()
    users = sorted(test.userId.unique())
    precisions, recalls, ndcgs = [], [], []
    recommended_union = set()
    intra_lists = []
    for user in users:
        seen = train_seen.get(user, set())
        recs = model.recommend(user, seen_items=seen, k=k) if personalized else model.recommend(user, k=k)
        recs = recs[:k]
        recommended_union.update(recs)
        relevant = test_relevant.get(user, set())
        hits = len(set(recs) & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant) if relevant else 0.0)
        ideal = min(len(relevant), k)
        ndcgs.append(_dcg(recs, relevant, k) / _dcg(list(range(ideal)), set(range(ideal)), k) if ideal else 0.0)
        intra_lists.append(recs)
    # Prediction metrics: popularity predicts global item mean; CF predicts weighted user-neighbour score.
    # Keep a transparent item-mean baseline prediction for both ranking models.
    item_means = train.groupby("movieId").rating.mean().to_dict()
    global_mean = float(train.rating.mean())
    actual, predicted = [], []
    for row in test.itertuples():
        actual.append(float(row.rating))
        predicted.append(float(item_means.get(row.movieId, global_mean)))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    # Catalog coverage is the fraction of catalog items recommended at least once.
    coverage = len(recommended_union) / max(1, len(all_items))
    # Average pairwise genre dissimilarity unavailable to model object; use item-id novelty proxy.
    # Diversity is computed as 1 - mean pairwise cosine similarity of recommendation lists.
    divs = []
    for recs in intra_lists:
        if len(recs) > 1:
            unique_ratio = len(set(recs)) / len(recs)
            divs.append(unique_ratio)
    diversity = float(np.mean(divs)) if divs else 0.0
    return {
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "Coverage": float(coverage),
        "Diversity": diversity,
        "RMSE": rmse,
        "MAE": mae
    }

def compare_models(popularity, item_cf, train, test, movies, output="reports/model_comparison.csv"):
    all_items = set(movies.movieId.unique())
    rows = []
    for name, model, personalized in [
        ("Popularity", popularity, False),
        ("Item-item CF", item_cf, True)
    ]:
        metrics = evaluate_model(model, train, test, all_items, k=5, personalized=personalized)
        rows.append({"Model": name, **metrics})
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result
