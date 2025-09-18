"""
Similarity search and recommendation engine
@Author Utkarsh Sharma
"""
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
from pathlib import Path
import config

class SimilaritySearch:
    """
    Handles similarity search and recommendations
    """

    def __init__(self, features=None, use_faiss=True):
        """
        Initialize similarity search

        Args:
            features: Feature matrix (n_items x feature_dim)
            use_faiss: Whether to use FAISS for fast search
        """
        self.features = features
        self.use_faiss = use_faiss
        self.index = None

        if features is not None:
            self.build_index(features)

    def build_index(self, features):
        """
        Build search index
        """
        self.features = normalize(features, norm='l2')

        if self.use_faiss:
            # Build FAISS index for fast similarity search
            feature_dim = features.shape[1]

            # Use IndexFlatIP for inner product (equivalent to cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(feature_dim)

            # Add vectors to index
            self.index.add(self.features.astype('float32'))

        print(f"Built index with {len(features)} items")

    def search_similar(self, query_features, k=10):
        """
        Search for similar items

        Args:
            query_features: Query feature vector(s)
            k: Number of similar items to return

        Returns:
            distances: Similarity scores
            indices: Indices of similar items
        """
        # Normalize query features
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)
        query_features = normalize(query_features, norm='l2')

        if self.use_faiss and self.index is not None:
            # Use FAISS for search
            distances, indices = self.index.search(
                query_features.astype('float32'), k
            )
        else:
            # Use sklearn cosine similarity
            similarities = cosine_similarity(query_features, self.features)

            # Get top-k indices
            indices = np.argsort(similarities[0])[::-1][:k]
            distances = similarities[0][indices]

            indices = indices.reshape(1, -1)
            distances = distances.reshape(1, -1)

        return distances, indices

    def get_recommendations(self, item_idx, k=10, exclude_self=True):
        """
        Get recommendations based on a specific item

        Args:
            item_idx: Index of the query item
            k: Number of recommendations
            exclude_self: Whether to exclude the query item from results

        Returns:
            indices: Indices of recommended items
            scores: Similarity scores
        """
        # Get query features
        query_features = self.features[item_idx]

        # Search for similar items
        n_search = k + 1 if exclude_self else k
        distances, indices = self.search_similar(query_features, n_search)

        # Process results
        indices = indices[0]
        distances = distances[0]

        if exclude_self:
            # Remove the query item itself
            mask = indices != item_idx
            indices = indices[mask][:k]
            distances = distances[mask][:k]

        return indices, distances

    def get_diverse_recommendations(self, item_idx, k=10, diversity_weight=0.3):
        """
        Get diverse recommendations using MMR (Maximal Marginal Relevance)

        Args:
            item_idx: Index of the query item
            k: Number of recommendations
            diversity_weight: Weight for diversity (0=pure similarity, 1=pure diversity)

        Returns:
            indices: Indices of recommended items
            scores: Relevance scores
        """
        # Get initial candidates (3x the required number)
        candidates_idx, candidates_sim = self.get_recommendations(
            item_idx, k=min(k*3, len(self.features)-1), exclude_self=True
        )

        # MMR selection
        selected = []
        selected_features = []
        remaining = list(range(len(candidates_idx)))

        # Select first item (most similar)
        selected.append(candidates_idx[0])
        selected_features.append(self.features[candidates_idx[0]])
        remaining.remove(0)

        # Iteratively select diverse items
        while len(selected) < k and remaining:
            scores = []

            for idx in remaining:
                item_feat = self.features[candidates_idx[idx]]

                # Similarity to query
                sim_to_query = candidates_sim[idx]

                # Maximum similarity to already selected items
                if selected_features:
                    sims_to_selected = [
                        cosine_similarity(
                            item_feat.reshape(1, -1),
                            feat.reshape(1, -1)
                        )[0, 0]
                        for feat in selected_features
                    ]
                    max_sim_to_selected = max(sims_to_selected)
                else:
                    max_sim_to_selected = 0

                # MMR score
                mmr_score = (1 - diversity_weight) * sim_to_query - \
                           diversity_weight * max_sim_to_selected
                scores.append(mmr_score)

            # Select item with highest MMR score
            best_idx = remaining[np.argmax(scores)]
            selected.append(candidates_idx[best_idx])
            selected_features.append(self.features[candidates_idx[best_idx]])
            remaining.remove(best_idx)

        return np.array(selected), np.ones(len(selected))

    def find_complementary_items(self, item_idx, data_loader, k=10):
        """
        Find complementary items (different category but similar style)

        Args:
            item_idx: Index of the query item
            data_loader: Data loader with metadata
            k: Number of complementary items

        Returns:
            indices: Indices of complementary items
            scores: Relevance scores
        """
        # Get query item info
        query_info = data_loader.get_product_info(item_idx)
        query_category = query_info['masterCategory']
        query_gender = query_info['gender']

        # Get similar items
        similar_idx, similar_scores = self.get_recommendations(
            item_idx, k=min(k*5, len(self.features)-1), exclude_self=True
        )

        # Filter for different categories but same gender
        complementary = []
        for idx, score in zip(similar_idx, similar_scores):
            item_info = data_loader.get_product_info(idx)

            if (item_info['masterCategory'] != query_category and
                item_info['gender'] == query_gender):
                complementary.append((idx, score))

            if len(complementary) >= k:
                break

        if complementary:
            indices, scores = zip(*complementary)
            return np.array(indices), np.array(scores)
        else:
            return np.array([]), np.array([])

    def find_outfit_items(self, item_idx, data_loader, k=5):
        """
        Find items to complete an outfit

        Args:
            item_idx: Index of the query item
            data_loader: Data loader with metadata
            k: Number of outfit items

        Returns:
            outfit_items: Dictionary of category -> (indices, scores)
        """
        # Get query item info
        query_info = data_loader.get_product_info(item_idx)
        query_category = query_info['masterCategory']
        query_gender = query_info['gender']
        query_color = query_info.get('baseColour', 'Multi')

        # Define outfit categories based on query item
        outfit_categories = {
            'Apparel': ['Footwear', 'Accessories'],
            'Footwear': ['Apparel', 'Accessories'],
            'Accessories': ['Apparel', 'Footwear']
        }

        target_categories = outfit_categories.get(query_category, ['Apparel', 'Footwear', 'Accessories'])

        # Get similar items
        similar_idx, similar_scores = self.get_recommendations(
            item_idx, k=min(k*10, len(self.features)-1), exclude_self=True
        )

        outfit_items = {}

        for target_cat in target_categories:
            cat_items = []
            for idx, score in zip(similar_idx, similar_scores):
                item_info = data_loader.get_product_info(idx)

                # Check if item matches criteria
                if (item_info['masterCategory'] == target_cat and
                    item_info['gender'] == query_gender):

                    # Bonus for matching or complementary colors
                    color_bonus = 0
                    if item_info.get('baseColour') == query_color:
                        color_bonus = 0.1
                    elif item_info.get('baseColour') in ['Black', 'White', 'Grey']:
                        color_bonus = 0.05

                    cat_items.append((idx, score + color_bonus))

                if len(cat_items) >= 2:  # Get 2 items per category
                    break

            if cat_items:
                indices, scores = zip(*cat_items)
                outfit_items[target_cat] = (np.array(indices), np.array(scores))

        return outfit_items

    def save_index(self, save_path):
        """
        Save search index
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'features': self.features,
                'index': self.index if not self.use_faiss else None
            }, f)

        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, str(save_path.with_suffix('.faiss')))

    def load_index(self, load_path):
        """
        Load search index
        """
        load_path = Path(load_path)

        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            self.features = data['features']

        if self.use_faiss:
            faiss_path = load_path.with_suffix('.faiss')
            if faiss_path.exists():
                self.index = faiss.read_index(str(faiss_path))
