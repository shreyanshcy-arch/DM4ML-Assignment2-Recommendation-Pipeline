from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class PopularityRecommender:
    """Recommend items using rating count with mean-rating tie breaking."""
    def fit(self, train):
        stats = train.groupby("movieId").agg(
            count=("rating", "count"), mean_rating=("rating", "mean")
        )
        stats["score"] = stats["mean_rating"] * np.log1p(stats["count"])
        self.ranking = stats.sort_values(["score", "count"], ascending=False).index.tolist()
        self.item_scores = stats["score"].to_dict()
        self.known_items = set(train["movieId"].unique())
        return self

    def recommend(self, user_id, seen_items=None, k=5):
        seen = set(seen_items or [])
        return [int(x) for x in self.ranking if x not in seen][:k]

class ItemItemCFRecommender:
    """Item-item collaborative filtering using cosine similarity of user vectors."""
    def fit(self, train):
        self.user_ids = sorted(train.userId.unique())
        self.item_ids = sorted(train.movieId.unique())
        self.user_index = {u:i for i,u in enumerate(self.user_ids)}
        self.item_index = {m:i for i,m in enumerate(self.item_ids)}
        rows = train.userId.map(self.user_index).to_numpy()
        cols = train.movieId.map(self.item_index).to_numpy()
        vals = train.rating.to_numpy(dtype=float)
        self.matrix = csr_matrix((vals, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids)))
        self.similarity = cosine_similarity(self.matrix.T, dense_output=False)
        self.popularity = train.groupby("movieId").rating.count().to_dict()
        return self

    def recommend(self, user_id, seen_items=None, k=5):
        if user_id not in self.user_index:
            return sorted(self.item_ids, key=lambda m: self.popularity.get(m, 0), reverse=True)[:k]
        seen = set(seen_items or [])
        urow = self.matrix.getrow(self.user_index[user_id]).toarray().ravel()
        scores = urow @ self.similarity
        candidates = []
        for movie_id, idx in self.item_index.items():
            if movie_id not in seen:
                candidates.append((movie_id, float(scores[idx])))
        candidates.sort(key=lambda x: (x[1], self.popularity.get(x[0], 0)), reverse=True)
        return [int(m) for m,_ in candidates[:k]]

def train_models(train, model_dir="models"):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    popularity = PopularityRecommender().fit(train)
    item_cf = ItemItemCFRecommender().fit(train)
    joblib.dump(popularity, Path(model_dir) / "popularity_model.joblib")
    joblib.dump(item_cf, Path(model_dir) / "item_item_cf_model.joblib")
    return popularity, item_cf
