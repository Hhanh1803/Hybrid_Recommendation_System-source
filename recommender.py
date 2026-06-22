import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommender:
    def __init__(self, products: pd.DataFrame, interactions: pd.DataFrame):
        self.products = products.copy()
        self.interactions = interactions.copy()
        self.products['description'] = self.products['description'].fillna('').astype(str)
        self._build_content_model()
        self._build_interaction_models()

    def _build_content_model(self):
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.products['description'])
        self.content_sim = cosine_similarity(self.tfidf_matrix)
        self.product_index = {item_id: idx for idx, item_id in enumerate(self.products['item_id'])}
        self.index_product = {idx: item_id for item_id, idx in self.product_index.items()}

    def _build_interaction_models(self):
        self.interaction_matrix = self.interactions.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        self.user_ids = list(self.interaction_matrix.index)
        self.item_ids = list(self.interaction_matrix.columns)
        self.user_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_index_interaction = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

        self.user_similarity = cosine_similarity(self.interaction_matrix)
        self.item_similarity = cosine_similarity(self.interaction_matrix.T)

        self.item_cooccurrence = self.interaction_matrix.T.dot(self.interaction_matrix)
        self.item_cooccurrence = self.item_cooccurrence.astype(int)

    def train_collaborative(self):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.interactions[['user_id', 'item_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.svd = SVD()
        self.svd.fit(trainset)
        return 'SVD collaborative model trained successfully.'

    def predict_svd(self, user_id: str, item_id: str) -> float:
        try:
            prediction = self.svd.predict(user_id, item_id)
            return float(prediction.est)
        except Exception:
            return 0.0

    def predict_user_user(self, user_id: str, item_id: str) -> float | None:
        if user_id not in self.user_index or item_id not in self.item_index_interaction:
            return None
        sim_scores = self.user_similarity[self.user_index[user_id]]
        item_ratings = self.interaction_matrix[item_id]
        rated_users_mask = item_ratings > 0
        if not rated_users_mask.any():
            return None
        sims = sim_scores[rated_users_mask]
        ratings = item_ratings[rated_users_mask].values
        weighted_ratings = np.dot(sims, ratings)
        weight_sum = np.sum(np.abs(sims))
        return float(weighted_ratings / weight_sum) if weight_sum > 0 else None

    def predict_item_item(self, user_id: str, item_id: str) -> float | None:
        if user_id not in self.user_index or item_id not in self.item_index_interaction:
            return None
        user_ratings = self.interaction_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        if rated_items.empty:
            return None
        target_idx = self.item_index_interaction[item_id]
        rated_indices = [self.item_index_interaction[item] for item in rated_items.index if item in self.item_index_interaction]
        if not rated_indices:
            return None
        sims = self.item_similarity[target_idx, rated_indices]
        ratings = rated_items.values
        weighted_ratings = np.dot(sims, ratings)
        weight_sum = np.sum(np.abs(sims))
        return float(weighted_ratings / weight_sum) if weight_sum > 0 else None

    def _user_content_profile(self, user_id: str):
        if user_id not in self.user_index:
            return None
        items = self.interaction_matrix.loc[user_id]
        liked_items = [item for item, rating in items.items() if rating > 0 and item in self.product_index]
        if not liked_items:
            return None
        indices = [self.product_index[item] for item in liked_items]
        profile = self.tfidf_matrix[indices].mean(axis=0)
        if hasattr(profile, 'toarray'):
            profile = profile.toarray()
        else:
            profile = np.asarray(profile)
        return profile.reshape(1, -1)

    def _content_similarity(self, user_id: str, item_id: str) -> float:
        profile = self._user_content_profile(user_id)
        if profile is None or item_id not in self.product_index:
            return 0.0
        item_vec = self.tfidf_matrix[self.product_index[item_id]]
        if hasattr(item_vec, 'toarray'):
            item_vec = item_vec.toarray()
        else:
            item_vec = np.asarray(item_vec)
        item_vec = item_vec.reshape(1, -1)
        sim = cosine_similarity(profile, item_vec)
        return float(sim[0, 0])

    def similar_products(self, item_id: str, top_n: int = 10) -> list[tuple[str, float]]:
        if item_id not in self.product_index:
            return []
        idx = self.product_index[item_id]
        scores = list(enumerate(self.content_sim[idx]))
        scores = sorted(scores, key=lambda pair: pair[1], reverse=True)
        results = []
        for other_idx, score in scores:
            if other_idx == idx:
                continue
            results.append((self.index_product[other_idx], float(score)))
            if len(results) >= top_n:
                break
        return results

    def frequently_bought_together(self, item_id: str, top_n: int = 10) -> list[tuple[str, int]]:
        if item_id not in self.item_cooccurrence.index:
            return []
        counts = self.item_cooccurrence.loc[item_id].copy()
        counts = counts.drop(index=item_id, errors='ignore')
        counts = counts[counts > 0].sort_values(ascending=False)
        return list(counts.head(top_n).items())

    def personalized_recommendations(
        self,
        user_id: str,
        top_n: int = 10,
        svd_w: float = 0.55,
        item_w: float = 0.30,
        content_w: float = 0.15
    ) -> list[tuple[str, float, list[str]]]:
        if user_id not in self.user_ids:
            return []
        interacted = set(self.interaction_matrix.loc[user_id].loc[lambda x: x > 0].index)
        candidates = [item for item in self.item_ids if item not in interacted]
        if not candidates:
            return []

        # Vectorized Item-Item predictions for all items at once
        user_ratings = self.interaction_matrix.loc[user_id]
        rated_mask = user_ratings > 0
        if rated_mask.any():
            rated_items = user_ratings[rated_mask]
            rated_indices = [self.item_index_interaction[item] for item in rated_items.index]
            sims = self.item_similarity[:, rated_indices]
            ratings = rated_items.values
            weighted_ratings = sims.dot(ratings)
            weight_sum = np.abs(sims).sum(axis=1)
            pred_ratings = np.zeros_like(weighted_ratings)
            valid_mask = weight_sum > 0
            pred_ratings[valid_mask] = weighted_ratings[valid_mask] / weight_sum[valid_mask]
            item_score_map = {self.item_ids[idx]: float(pred_ratings[idx]) for idx in range(len(self.item_ids))}
        else:
            item_score_map = {}

        # Vectorized content similarity computation
        profile = self._user_content_profile(user_id)
        if profile is not None:
            content_sims = cosine_similarity(profile, self.tfidf_matrix).flatten()
            content_score_map = {
                item_id: float(content_sims[self.product_index[item_id]])
                for item_id in candidates if item_id in self.product_index
            }
        else:
            content_score_map = {}

        scores = []
        for item_id in candidates:
            svd_score = self.predict_svd(user_id, item_id)
            item_score = item_score_map.get(item_id, 0.0)
            content_score = content_score_map.get(item_id, 0.0)

            final_score = svd_w * svd_score + item_w * item_score + content_w * (content_score * 5)

            reasons = []
            if svd_w > 0 and svd_score >= 3.5:
                reasons.append("Sở thích tương đồng với nhóm khách hàng chung")
            if item_w > 0 and item_score >= 3.0:
                reasons.append("Tương tự sản phẩm bạn đã từng xem/mua")
            if content_w > 0 and content_score >= 0.2:
                reasons.append("Mô tả phù hợp với danh mục yêu thích")
            if not reasons:
                reasons.append("Gợi ý tổng hợp dựa trên xu hướng")

            scores.append((item_id, final_score, reasons))

        scores = sorted(scores, key=lambda pair: pair[1], reverse=True)
        return scores[:top_n]

    def trending_products(self, top_n: int = 10) -> list[tuple[str, int]]:
        popularity = self.interactions['item_id'].value_counts()
        return list(popularity.head(top_n).items())

    def cart_recommendations(self, cart_item_ids: list[str], top_n: int = 10) -> list[tuple[str, float]]:
        score_map = {}
        for item_id in cart_item_ids:
            for other_id, count in self.frequently_bought_together(item_id, top_n * 2):
                if other_id in cart_item_ids:
                    continue
                score_map[other_id] = score_map.get(other_id, 0.0) + count
            for other_id, sim in self.similar_products(item_id, top_n * 2):
                if other_id in cart_item_ids:
                    continue
                score_map[other_id] = score_map.get(other_id, 0.0) + sim
        scores = sorted(score_map.items(), key=lambda pair: pair[1], reverse=True)
        return scores[:top_n]

    def add_interaction(self, user_id: str, item_id: str, rating: float):
        # Update interactions dataframe
        mask = (self.interactions['user_id'] == user_id) & (self.interactions['item_id'] == item_id)
        if mask.any():
            self.interactions.loc[mask, 'rating'] = rating
        else:
            new_row = pd.DataFrame([{'user_id': user_id, 'item_id': item_id, 'rating': rating}])
            self.interactions = pd.concat([self.interactions, new_row], ignore_index=True)

        # Update product catalog index if new item is in products df but not currently indexed
        if item_id not in self.product_index and item_id in self.products['item_id'].values:
            self._build_content_model()

        # Re-build interaction metrics
        self._build_interaction_models()

        # Re-train SVD CF
        self.train_collaborative()
