"""
Advanced Hybrid Recommendation Engine combining multiple approaches:
1. Content-Based Filtering (TF-IDF + KNN)
2. Collaborative Filtering (Matrix Factorization)
3. User Preference Learning
4. Demographic Filtering
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix
from scipy.spatial.distance import cosine
import joblib
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import json
import time
import shutil
import tempfile

logger = logging.getLogger(__name__)

# Model version for compatibility checking
MODEL_VERSION = "2.0.0"


class HybridManwhaRecommender:
    """
    Hybrid recommendation system that combines:
    - Content-based filtering (TF-IDF on descriptions, tags, genres)
    - Genre-based similarity (genre co-occurrence patterns via SVD)
    - User preferences (personalized filtering)

    Note: This does NOT include true collaborative filtering, which would require
    user-item interaction data (ratings, views, etc.). The genre similarity uses
    item-item relationships based on shared genres.
    """

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 tfidf_params: Optional[Dict] = None,
                 svd_params: Optional[Dict] = None,
                 knn_params: Optional[Dict] = None):
        """
        Initialize Hybrid Recommender with configurable hyperparameters.

        Args:
            weights: Component weights for hybrid scoring
            tfidf_params: TF-IDF vectorizer parameters
            svd_params: SVD dimensionality reduction parameters
            knn_params: KNN model parameters
        """
        self.df = None
        self.content_model = None
        self.genre_model = None
        self.tfidf_vectorizer = None
        self.rating_scaler = None
        self.popularity_scaler = None  # Save as instance variable
        self.feature_matrix = None
        self.user_preferences = {}

        # Configurable weights
        self.weights = weights or {
            'content': 0.4,
            'genre_similarity': 0.3,
            'user_pref': 0.3
        }

        # Configurable TF-IDF parameters
        self.tfidf_params = tfidf_params or {
            'max_features': 5000,
            'stop_words': None,  # Removed 'english' - wrong for Korean content
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.8
        }

        # Configurable SVD parameters
        self.svd_params = svd_params or {
            'explained_variance_threshold': 0.90  # Auto-select components
        }

        # Configurable KNN parameters
        self.knn_params = knn_params or {
            'metric': 'cosine',
            'algorithm': 'brute'
        }

        # Store all hyperparameters
        self.hyperparameters = {
            'weights': self.weights,
            'tfidf': self.tfidf_params,
            'svd': self.svd_params,
            'knn': self.knn_params
        }

        # Model version
        self.model_version = MODEL_VERSION

        # Cache and tracking variables
        self._title_cache = None
        self._popularity_ranks = None
        self._all_recommended_items = set()

    def prepare_data(self, catalog_path: str) -> pd.DataFrame:
        """Load and prepare manhwa catalog data with comprehensive validation."""
        logger.info(f"Loading catalog from {catalog_path}")

        # Validate file exists
        catalog_file = Path(catalog_path)
        if not catalog_file.exists():
            raise FileNotFoundError(f"Catalog file not found: {catalog_path}")
        if not catalog_file.is_file():
            raise ValueError(f"Path is not a file: {catalog_path}")

        # Load and validate JSON
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in catalog file: {e}")
        except PermissionError:
            raise PermissionError(f"No read permission for: {catalog_path}")

        # Validate data structure
        if not isinstance(data, list):
            raise ValueError(f"Catalog must be a list, got {type(data)}")

        # Check minimum data requirements
        if len(data) < 1:
            raise ValueError(f"Insufficient data: {len(data)} entries. Need at least 1 entry.")

        # Warn if data is too small for meaningful recommendations
        if len(data) < 20:
            logger.warning(f"Small dataset: {len(data)} entries. For meaningful recommendations, at least 20 entries recommended.")

        self.df = pd.DataFrame(data)

        # Validate required columns
        required_columns = ['name']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate data types
        for idx, row in self.df.iterrows():
            if not isinstance(row.get('name'), str) or not row.get('name').strip():
                raise ValueError(f"Entry {idx}: 'name' must be non-empty string")
            if 'description' in row and row['description'] is not None and not isinstance(row['description'], str):
                raise ValueError(f"Entry {idx}: 'description' must be string or null")
            if 'genres' in row and row['genres'] is not None and not isinstance(row.get('genres'), list):
                raise ValueError(f"Entry {idx}: 'genres' must be a list")

        # Check for duplicates
        duplicates = self.df[self.df.duplicated(subset=['name'], keep=False)]
        if not duplicates.empty:
            dup_names = duplicates['name'].tolist()
            logger.warning(f"Found {len(duplicates)} duplicate entries: {dup_names[:5]}...")
            self.df = self.df.drop_duplicates(subset=['name'], keep='first')

        # Handle missing values
        if 'description' not in self.df.columns:
            self.df['description'] = ''
        else:
            self.df['description'] = self.df['description'].fillna('')

        # Add rating if it doesn't exist
        if 'rating' not in self.df.columns:
            self.df['rating'] = 3.5  # Default median rating
        else:
            median_rating = self.df['rating'].median()
            self.df['rating'] = self.df['rating'].fillna(median_rating if pd.notna(median_rating) else 3.5)

        # Add popularity if it doesn't exist
        if 'popularity' not in self.df.columns:
            self.df['popularity'] = 0
        else:
            self.df['popularity'] = self.df['popularity'].fillna(0)

        # Handle genres/tags - older data uses 'tags', newer uses both
        if 'genres' not in self.df.columns and 'tags' in self.df.columns:
            # Use tags as genres for older data
            self.df['genres'] = self.df['tags']
            self.df['tags'] = self.df['tags']
        elif 'genres' not in self.df.columns:
            self.df['genres'] = [[] for _ in range(len(self.df))]

        if 'tags' not in self.df.columns:
            self.df['tags'] = self.df['genres']

        self.df['genres'] = self.df['genres'].apply(lambda x: x if isinstance(x, list) else [])
        self.df['tags'] = self.df['tags'].apply(lambda x: x if isinstance(x, list) else [])

        logger.info(f"Loaded {len(self.df)} manhwa entries")

        return self.df

    def create_evaluation_split(self, test_ratio: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets for evaluation.

        Args:
            test_ratio: Proportion of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, test_df)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call prepare_data() first.")

        # Shuffle and split
        shuffled_df = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(shuffled_df) * (1 - test_ratio))

        train_df = shuffled_df.iloc[:split_idx].copy()
        test_df = shuffled_df.iloc[split_idx:].copy()

        logger.info(f"Created train/test split: {len(train_df)} train, {len(test_df)} test")

        return train_df, test_df

    def evaluate_recommendations(self, test_df: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        """
        Evaluate recommendation quality using standard metrics plus coverage, novelty, diversity.

        For each manhwa in test set, generate recommendations and measure:
        - Precision@K: How many of top K recs share genres with the test item
        - Recall@K: Of all items that share genres, how many are in top K
        - NDCG@K: Normalized Discounted Cumulative Gain
        - MRR: Mean Reciprocal Rank
        - Hit Rate@K: Percentage where at least 1 relevant item in top K
        - Coverage: Percentage of catalog items recommended
        - Novelty: Average popularity rank (higher = more novel)
        - Diversity: Average intra-list diversity

        Args:
            test_df: Test set dataframe
            k: Number of recommendations to evaluate

        Returns:
            Dictionary of metric scores
        """
        if self.content_model is None:
            raise ValueError("Model not trained. Call build_content_features() first.")

        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        mrr_scores = []
        hits = 0
        all_recommendation_lists = []

        logger.info(f"Evaluating on {len(test_df)} test items with K={k}...")

        for idx, row in test_df.iterrows():
            test_title = row['name']
            test_genres = set(row['genres']) if isinstance(row['genres'], list) else set()

            # Skip if test item not in training data or has no genres
            if test_title not in self.df['name'].values or len(test_genres) == 0:
                continue

            try:
                # Get recommendations
                recs = self.recommend(test_title, n_recommendations=k)
                all_recommendation_lists.append(recs)

                # Track all recommended items for coverage
                for rec in recs:
                    self._all_recommended_items.add(rec['name'])

                # Calculate relevance: items that share at least one genre
                relevant_items = []
                ranks = []
                dcg = 0.0

                for rank, rec in enumerate(recs, 1):
                    rec_genres = set(rec['genres']) if isinstance(rec['genres'], list) else set()
                    is_relevant = len(test_genres & rec_genres) > 0

                    if is_relevant:
                        relevant_items.append(rec)
                        if len(ranks) == 0:  # First relevant item
                            mrr_scores.append(1.0 / rank)
                        ranks.append(rank)

                        # NDCG: relevance score is Jaccard similarity of genres
                        relevance = len(test_genres & rec_genres) / len(test_genres | rec_genres)
                        dcg += relevance / np.log2(rank + 1)

                # Precision@K
                precision = len(relevant_items) / k if k > 0 else 0
                precision_scores.append(precision)

                # Recall@K (of all items in training that share genres, how many in top K)
                all_relevant = self.df[
                    self.df.apply(
                        lambda x: len(test_genres & set(x['genres'] if isinstance(x['genres'], list) else [])) > 0,
                        axis=1
                    ) & (self.df['name'] != test_title)
                ]
                recall = len(relevant_items) / len(all_relevant) if len(all_relevant) > 0 else 0
                recall_scores.append(recall)

                # NDCG@K - normalize by ideal DCG
                ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(all_relevant))))
                ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
                ndcg_scores.append(ndcg)

                # Hit rate
                if len(relevant_items) > 0:
                    hits += 1

            except Exception as e:
                logger.warning(f"Error evaluating {test_title}: {e}")
                continue

        # Add coverage metric
        coverage = len(self._all_recommended_items) / len(self.df) if len(self.df) > 0 else 0

        # Add novelty metric (average popularity rank)
        novelty_scores = []
        if self._popularity_ranks is None:
            self._popularity_ranks = self.df['popularity'].rank(ascending=False).to_dict()

        all_recommendations = [rec for recs in all_recommendation_lists for rec in recs]
        for rec in all_recommendations:
            idx = self.df[self.df['name'] == rec['name']].index
            if len(idx) > 0:
                rank = self._popularity_ranks.get(idx[0], len(self.df))
                novelty_scores.append(rank / len(self.df))  # Normalize

        novelty = np.mean(novelty_scores) if novelty_scores else 0

        # Add diversity metric
        diversity_scores = []
        for recs in all_recommendation_lists:
            diversity_scores.append(self.calculate_diversity(recs))

        diversity = np.mean(diversity_scores) if diversity_scores else 0

        # Calculate final metrics
        metrics = {
            'precision@k': np.mean(precision_scores) if precision_scores else 0.0,
            'recall@k': np.mean(recall_scores) if recall_scores else 0.0,
            'ndcg@k': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'hit_rate@k': hits / len(test_df) if len(test_df) > 0 else 0.0,
            'coverage': coverage,
            'novelty': novelty,
            'diversity': diversity,
            'k': k,
            'n_test_items': len(test_df),
            'n_evaluated': len(precision_scores)
        }

        logger.info(f"Evaluation Results (K={k}):")
        logger.info(f"  Precision@{k}: {metrics['precision@k']:.4f}")
        logger.info(f"  Recall@{k}: {metrics['recall@k']:.4f}")
        logger.info(f"  NDCG@{k}: {metrics['ndcg@k']:.4f}")
        logger.info(f"  MRR: {metrics['mrr']:.4f}")
        logger.info(f"  Hit Rate@{k}: {metrics['hit_rate@k']:.4f}")
        logger.info(f"  Coverage: {metrics['coverage']:.4f}")
        logger.info(f"  Novelty: {metrics['novelty']:.4f}")
        logger.info(f"  Diversity: {metrics['diversity']:.4f}")

        return metrics

    def build_content_features(self):
        """Build content-based features using TF-IDF and metadata."""
        logger.info("Building content-based features...")

        # Combine text fields for TF-IDF using vectorized operations (much faster than iterrows)
        text_features = (
            self.df['description'].fillna('') + ' ' +
            self.df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' ' +
            self.df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        ).tolist()

        # TF-IDF vectorization with configurable parameters
        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_params)

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Normalize ratings
        self.rating_scaler = MinMaxScaler()
        scaled_ratings = self.rating_scaler.fit_transform(
            self.df[['rating']].values
        )

        # Normalize popularity and save scaler
        self.popularity_scaler = MinMaxScaler()
        scaled_popularity = self.popularity_scaler.fit_transform(
            self.df[['popularity']].values
        )

        # Combine features
        self.feature_matrix = hstack([
            tfidf_matrix,
            csr_matrix(scaled_ratings),
            csr_matrix(scaled_popularity)
        ])

        logger.info(f"Combined feature matrix shape: {self.feature_matrix.shape}")

        # Train KNN model with configurable parameters
        self.content_model = NearestNeighbors(
            n_neighbors=min(21, len(self.df)),  # Top 20 + self
            metric=self.knn_params['metric'],
            algorithm=self.knn_params['algorithm']
        )
        self.content_model.fit(self.feature_matrix)

        logger.info("Content-based model trained successfully")

    def build_genre_similarity_features(self):
        """
        Build genre-based similarity features using SVD with automatic component selection.

        This creates a lower-dimensional representation of manhwa based on their genres,
        allowing us to find similar items based on genre patterns. This is content-based
        filtering, not collaborative filtering (which would require user interaction data).
        """
        logger.info("Building genre-based similarity features...")

        # Create genre-based profiles (binary matrix of genre presence)
        genre_profiles = self._create_genre_profiles()

        # Apply dimensionality reduction with auto component selection
        if len(genre_profiles) > 0:
            # Determine optimal n_components based on explained variance
            max_components = min(100, len(genre_profiles) - 1)
            temp_svd = TruncatedSVD(n_components=max_components, random_state=42)
            temp_svd.fit(genre_profiles)

            explained_var = temp_svd.explained_variance_ratio_.cumsum()
            threshold = self.svd_params.get('explained_variance_threshold', 0.90)
            n_components = np.argmax(explained_var >= threshold) + 1

            logger.info(f"Using {n_components} components to explain {threshold*100}% variance")

            # Final SVD with optimal components
            self.genre_model = TruncatedSVD(n_components=n_components, random_state=42)
            self.genre_features = self.genre_model.fit_transform(genre_profiles)

            logger.info(f"Genre similarity features shape: {self.genre_features.shape}")
        else:
            self.genre_features = None

    def _create_genre_profiles(self) -> np.ndarray:
        """
        Create binary genre matrix using efficient sklearn MultiLabelBinarizer.
        Each row represents a manhwa, each column a genre (1 if present, 0 otherwise).
        """
        from sklearn.preprocessing import MultiLabelBinarizer

        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(self.df['genres'])

        logger.info(f"Created genre matrix: {genre_matrix.shape} ({mlb.classes_.size} unique genres)")

        return genre_matrix

    def tune_hyperparameters(self,
                            train_df: pd.DataFrame,
                            val_df: pd.DataFrame,
                            param_grid: Optional[Dict] = None,
                            metric: str = 'ndcg@k',
                            k: int = 10) -> Dict:
        """
        Tune hyperparameters using grid search.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            param_grid: Parameter grid to search. If None, uses defaults.
            metric: Metric to optimize ('ndcg@k', 'precision@k', 'mrr')
            k: K value for ranking metrics

        Returns:
            Dict with best_params, best_score, all_results
        """
        if param_grid is None:
            # Default search space
            param_grid = {
                'weights': [
                    {'content': 0.5, 'genre_similarity': 0.3, 'user_pref': 0.2},
                    {'content': 0.4, 'genre_similarity': 0.3, 'user_pref': 0.3},
                    {'content': 0.3, 'genre_similarity': 0.4, 'user_pref': 0.3},
                    {'content': 0.4, 'genre_similarity': 0.4, 'user_pref': 0.2},
                ],
                'tfidf_max_features': [3000, 5000, 10000],
                'tfidf_min_df': [1, 2, 3],
                'tfidf_max_df': [0.7, 0.8, 0.9]
            }

        logger.info(f"Starting hyperparameter tuning, optimizing for {metric}@{k}")

        best_score = 0
        best_params = None
        all_results = []

        from itertools import product

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        for combination in product(*values):
            params = dict(zip(keys, combination))

            # Apply parameters
            if 'weights' in params:
                self.weights = params['weights']
            if 'tfidf_max_features' in params:
                self.tfidf_params['max_features'] = params['tfidf_max_features']
            if 'tfidf_min_df' in params:
                self.tfidf_params['min_df'] = params['tfidf_min_df']
            if 'tfidf_max_df' in params:
                self.tfidf_params['max_df'] = params['tfidf_max_df']

            # Train on training set
            self.df = train_df
            self.build_content_features()
            self.build_genre_similarity_features()

            # Evaluate on validation set
            # Need to include val items in df for lookup
            self.df = pd.concat([train_df, val_df])

            try:
                metrics = self.evaluate_recommendations(val_df, k=k)
                score = metrics.get(metric, 0)

                result = {
                    'params': params.copy(),
                    'score': score,
                    'metrics': metrics
                }
                all_results.append(result)

                logger.info(f"Params: {params} -> {metric}: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception as e:
                logger.warning(f"Failed to evaluate params {params}: {e}")

        logger.info(f"Best {metric}: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")

        # Apply best parameters
        if best_params:
            if 'weights' in best_params:
                self.weights = best_params['weights']
            if 'tfidf_max_features' in best_params:
                self.tfidf_params['max_features'] = best_params['tfidf_max_features']
            if 'tfidf_min_df' in best_params:
                self.tfidf_params['min_df'] = best_params['tfidf_min_df']
            if 'tfidf_max_df' in best_params:
                self.tfidf_params['max_df'] = best_params['tfidf_max_df']

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results
        }

    def handle_cold_start(self,
                         user_profile: Optional[Dict] = None,
                         n_recommendations: int = 10,
                         popularity_bias: float = 0.5) -> List[Dict]:
        """
        Handle cold start scenarios for new users or when no history is available.

        Args:
            user_profile: Optional user preferences
            n_recommendations: Number of recommendations
            popularity_bias: 0-1, higher = more popular items (exploration vs exploitation)

        Returns:
            List of diverse, popular recommendations
        """
        if self.df is None or len(self.df) == 0:
            return []

        # Get top rated items
        top_rated = self.df.nlargest(n_recommendations * 3, 'rating')

        # Add popularity boost
        if 'popularity' in top_rated.columns:
            top_rated = top_rated.copy()
            top_rated['combined_score'] = (
                (1 - popularity_bias) * top_rated['rating'] +
                popularity_bias * (top_rated['popularity'] / top_rated['popularity'].max())
            )
            top_rated = top_rated.nlargest(n_recommendations * 2, 'combined_score')

        # Ensure genre diversity
        diverse_recs = []
        seen_genres = set()

        for _, row in top_rated.iterrows():
            if len(diverse_recs) >= n_recommendations:
                break

            row_genres = set(row.get('genres', []))
            # Add if it introduces new genres or we have few recs
            if len(diverse_recs) < 3 or len(row_genres - seen_genres) > 0:
                diverse_recs.append(row.to_dict())
                seen_genres.update(row_genres)

        # Fill remaining with top rated if needed
        while len(diverse_recs) < n_recommendations and len(diverse_recs) < len(top_rated):
            idx = len(diverse_recs)
            if idx < len(top_rated):
                diverse_recs.append(top_rated.iloc[idx].to_dict())

        return diverse_recs

    def calculate_diversity(self, recommendations: List[Dict]) -> float:
        """
        Calculate intra-list diversity of recommendations.

        Returns:
            Diversity score 0-1 (higher = more diverse)
        """
        if len(recommendations) < 2:
            return 1.0

        # Get feature vectors for each recommendation
        indices = []
        for rec in recommendations:
            idx = self.df[self.df['name'] == rec['name']].index
            if len(idx) > 0:
                indices.append(idx[0])

        if len(indices) < 2:
            return 1.0

        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                vec1 = self.feature_matrix[indices[i]].toarray().flatten()
                vec2 = self.feature_matrix[indices[j]].toarray().flatten()

                # Cosine distance = 1 - cosine similarity
                try:
                    dist = cosine(vec1, vec2)
                    if not np.isnan(dist):
                        similarities.append(1 - dist)  # Convert to similarity
                except:
                    pass

        if not similarities:
            return 1.0

        # Diversity = 1 - average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1 - avg_similarity

        return max(0, min(1, diversity))

    def _mmr_rerank(self,
                    candidate_scores: Dict[int, float],
                    query_idx: int,
                    n_recommendations: int,
                    diversity_weight: float = 0.5) -> List[int]:
        """
        Maximal Marginal Relevance re-ranking for diversity.

        Args:
            candidate_scores: Dict of {idx: relevance_score}
            query_idx: Query item index
            n_recommendations: Number to return
            diversity_weight: 0-1, higher = more diversity (lambda in MMR formula)

        Returns:
            List of reranked indices
        """
        if len(candidate_scores) <= n_recommendations:
            return list(candidate_scores.keys())

        selected = []
        remaining = set(candidate_scores.keys())

        # Select item with highest relevance first
        first_item = max(candidate_scores.items(), key=lambda x: x[1])[0]
        selected.append(first_item)
        remaining.remove(first_item)

        # Iteratively select items balancing relevance and diversity
        while len(selected) < n_recommendations and remaining:
            mmr_scores = {}

            for idx in remaining:
                # Relevance component
                relevance = candidate_scores[idx]

                # Diversity component: max similarity to already selected
                max_sim = 0
                for sel_idx in selected:
                    vec1 = self.feature_matrix[idx].toarray().flatten()
                    vec2 = self.feature_matrix[sel_idx].toarray().flatten()

                    try:
                        dist = cosine(vec1, vec2)
                        similarity = 1 - dist
                        max_sim = max(max_sim, similarity)
                    except:
                        pass

                # MMR = λ * Relevance - (1-λ) * MaxSimilarity
                mmr = diversity_weight * relevance - (1 - diversity_weight) * max_sim
                mmr_scores[idx] = mmr

            # Select item with highest MMR
            if mmr_scores:
                next_item = max(mmr_scores.items(), key=lambda x: x[1])[0]
                selected.append(next_item)
                remaining.remove(next_item)
            else:
                break

        return selected

    def _build_title_cache(self):
        """Build cache for O(1) title lookups."""
        self._title_cache = {
            title.lower(): idx
            for idx, title in enumerate(self.df['name'])
        }
    def get_content_recommendations(
        self,
        manhwa_title: str,
        n_recommendations: int = 20
    ) -> List[Tuple[int, float]]:
        """Get content-based recommendations."""
        # Find manhwa index
        idx = self._find_manhwa_index(manhwa_title)
        if idx is None:
            return []

        # Get neighbors
        distances, indices = self.content_model.kneighbors(
            self.feature_matrix[idx],
            n_neighbors=n_recommendations + 1
        )

        # Convert distances to similarities (1 - cosine distance)
        similarities = 1 - distances[0]

        # Return (index, score) pairs, excluding self
        results = [
            (int(indices[0][i]), float(similarities[i]))
            for i in range(1, len(indices[0]))
        ]

        return results

    def get_genre_similarity_recommendations(
        self,
        manhwa_title: str,
        n_recommendations: int = 20
    ) -> List[Tuple[int, float]]:
        """Get genre-based similarity recommendations using latent genre features."""
        if self.genre_features is None:
            return []

        idx = self._find_manhwa_index(manhwa_title)
        if idx is None:
            return []

        # Compute similarity in genre latent space
        target_features = self.genre_features[idx]
        similarities = np.dot(self.genre_features, target_features)

        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]

        results = [
            (int(i), float(similarities[i]))
            for i in top_indices
        ]

        return results

    # REVIEW: [HIGH] No bounds checking on manhwa_idx
    # Recommendation: Add validation that manhwa_idx is within DataFrame bounds
    # Location: get_user_preference_score function
    def get_user_preference_score(
        self,
        manhwa_idx: int,
        user_profile: Dict
    ) -> float:
        """
        Calculate preference score based on user's taste profile.

        User profile contains:
        - liked_genres: List of favorite genres
        - disliked_genres: List of disliked genres
        - min_rating: Minimum rating threshold
        - preferred_status: Preferred status (e.g., completed, ongoing)
        """
        # REVIEW: [MEDIUM] Magic number 0.5 as base score - should be documented or configurable
        # Location: Line 271
        score = 0.5  # Base score

        manhwa = self.df.iloc[manhwa_idx]

        # Genre preferences
        liked_genres = set(user_profile.get('liked_genres', []))
        disliked_genres = set(user_profile.get('disliked_genres', []))
        manhwa_genres = set(manhwa['genres'])

        # REVIEW: [LOW] Magic weights (0.3, 0.2, 0.1) not documented or tuned
        # Recommendation: Move to class constants with justification
        # Location: Lines 283-300
        # Boost for liked genres
        overlap = len(manhwa_genres & liked_genres)
        if liked_genres:
            score += 0.3 * (overlap / len(liked_genres))

        # Penalty for disliked genres
        if manhwa_genres & disliked_genres:
            score -= 0.3

        # Rating filter
        min_rating = user_profile.get('min_rating', 0)
        if manhwa['rating'] >= min_rating:
            score += 0.2
        else:
            score -= 0.3

        # Status preference
        preferred_status = user_profile.get('preferred_status', [])
        if preferred_status and manhwa.get('status') in preferred_status:
            score += 0.1

        # Normalize to 0-1
        score = max(0, min(1, score))

        return score

    def recommend(
        self,
        manhwa_title: str,
        n_recommendations: int = 10,
        user_profile: Optional[Dict] = None,
        filters: Optional[Dict] = None,
        diversity: float = 0.0
    ) -> List[Dict]:
        """
        Get hybrid recommendations with optional diversity re-ranking.

        Args:
            manhwa_title: Title of manhwa to base recommendations on
            n_recommendations: Number of recommendations to return
            user_profile: User preference profile
            filters: Additional filters (genre, rating, status, etc.)
            diversity: 0-1, higher = more diverse results (uses MMR)

        Returns:
            List of recommended manhwa with scores
        """
        # Find input manhwa index to exclude it from results
        input_idx = self._find_manhwa_index(manhwa_title)
        if input_idx is None:
            logger.warning(f"No match found for '{manhwa_title}'")
            return []

        # Get recommendations from each method
        # Request more candidates than needed to account for filtering
        # But not more than available in dataset
        max_candidates = min(50, len(self.df) - 1)
        n_candidates = min(n_recommendations * 5, max_candidates)

        content_recs = self.get_content_recommendations(manhwa_title, n_recommendations=n_candidates)
        genre_recs = self.get_genre_similarity_recommendations(manhwa_title, n_recommendations=n_candidates)

        # Combine scores
        combined_scores = {}

        # Content-based scores (TF-IDF similarity)
        for idx, score in content_recs:
            combined_scores[idx] = combined_scores.get(idx, 0) + self.weights['content'] * score

        # Genre-based similarity scores
        for idx, score in genre_recs:
            combined_scores[idx] = combined_scores.get(idx, 0) + self.weights['genre_similarity'] * score

        # User preference scores
        if user_profile:
            for idx in combined_scores.keys():
                pref_score = self.get_user_preference_score(idx, user_profile)
                combined_scores[idx] += self.weights['user_pref'] * pref_score

        # Apply filters
        if filters:
            combined_scores = self._apply_filters(combined_scores, filters)

        # Remove input manhwa from results (don't recommend what user already specified)
        if input_idx in combined_scores:
            del combined_scores[input_idx]

        # Apply MMR re-ranking if diversity requested
        if diversity > 0 and len(combined_scores) > n_recommendations:
            reranked_indices = self._mmr_rerank(
                combined_scores,
                input_idx,
                n_recommendations,
                diversity_weight=diversity
            )
            # Use reranked order
            sorted_recs = [(idx, combined_scores[idx]) for idx in reranked_indices]
        else:
            # Sort by combined score
            sorted_recs = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]

        # Build result list
        recommendations = []
        for idx, score in sorted_recs:
            manhwa = self.df.iloc[idx].to_dict()
            manhwa['recommendation_score'] = float(score)
            recommendations.append(manhwa)

        return recommendations

    def _apply_filters(self, scores: Dict[int, float], filters: Dict) -> Dict[int, float]:
        """Apply filters to recommendation candidates."""
        filtered_scores = {}

        for idx, score in scores.items():
            manhwa = self.df.iloc[idx]

            # Genre filter
            if 'genres' in filters:
                required_genres = set(filters['genres'])
                if not (set(manhwa['genres']) & required_genres):
                    continue

            # Rating filter
            if 'min_rating' in filters:
                if manhwa['rating'] < filters['min_rating']:
                    continue

            if 'max_rating' in filters:
                if manhwa['rating'] > filters['max_rating']:
                    continue

            # Status filter
            if 'status' in filters:
                if manhwa.get('status') not in filters['status']:
                    continue

            # Year filter
            if 'min_year' in filters:
                # TODO: Parse year from years field
                pass

            filtered_scores[idx] = score

        return filtered_scores

    def _find_manhwa_index(self, title: str) -> Optional[int]:
        """Find manhwa index by title with caching and fuzzy matching."""
        from rapidfuzz import fuzz, process

        # Build cache if needed
        if self._title_cache is None:
            self._build_title_cache()

        # Fast exact match using cache
        idx = self._title_cache.get(title.lower())
        if idx is not None:
            return idx

        # Fall back to fuzzy matching
        titles = self.df['name'].tolist()
        best_match = process.extractOne(
            title,
            titles,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=70
        )

        if best_match:
            matched_title, score, idx = best_match
            logger.info(f"Fuzzy matched '{title}' to '{matched_title}' (score: {score})")
            return idx

        logger.warning(f"No match found for '{title}'")
        return None

    def save_model(self, output_dir: str = "models"):
        """Save trained models with atomic writes and error handling."""
        # Validate model is trained
        if self.content_model is None:
            raise ValueError("Cannot save: content model not trained. Call build_content_features() first.")
        if self.df is None or len(self.df) == 0:
            raise ValueError("Cannot save: no data loaded. Call prepare_data() first.")

        output_path = Path(output_dir)

        # Check write permissions
        try:
            output_path.mkdir(exist_ok=True, parents=True)
        except PermissionError:
            raise PermissionError(f"No write permission for directory: {output_dir}")

        # Use temporary directory for atomic writes
        temp_dir = output_path / f".tmp_{int(time.time())}"

        try:
            temp_dir.mkdir(exist_ok=True)

            # Save each component with error handling
            try:
                joblib.dump(self.content_model, temp_dir / "content_model.pkl")
            except Exception as e:
                raise IOError(f"Failed to save content_model: {e}")

            try:
                joblib.dump(self.tfidf_vectorizer, temp_dir / "tfidf_vectorizer.pkl")
            except Exception as e:
                raise IOError(f"Failed to save tfidf_vectorizer: {e}")

            try:
                joblib.dump(self.rating_scaler, temp_dir / "rating_scaler.pkl")
            except Exception as e:
                raise IOError(f"Failed to save rating_scaler: {e}")

            try:
                joblib.dump(self.feature_matrix, temp_dir / "feature_matrix.pkl")
            except Exception as e:
                raise IOError(f"Failed to save feature_matrix: {e}")

            # Save popularity scaler if it exists
            if self.popularity_scaler is not None:
                try:
                    joblib.dump(self.popularity_scaler, temp_dir / "popularity_scaler.pkl")
                except Exception as e:
                    raise IOError(f"Failed to save popularity_scaler: {e}")

            # Save genre model if it exists
            if self.genre_model:
                try:
                    joblib.dump(self.genre_model, temp_dir / "genre_model.pkl")
                except Exception as e:
                    raise IOError(f"Failed to save genre_model: {e}")

                try:
                    joblib.dump(self.genre_features, temp_dir / "genre_features.pkl")
                except Exception as e:
                    raise IOError(f"Failed to save genre_features: {e}")

            # Save dataframe
            try:
                self.df.to_pickle(temp_dir / "manhwa_catalog.pkl")
            except Exception as e:
                raise IOError(f"Failed to save manhwa_catalog: {e}")

            # Save config
            config = {
                'version': self.model_version,
                'weights': self.weights,
                'hyperparameters': self.hyperparameters,
                'n_entries': len(self.df),
                'feature_matrix_shape': self.feature_matrix.shape,
                'has_genre_model': self.genre_model is not None,
                'timestamp': time.time()
            }

            try:
                with open(temp_dir / "recommender_config.json", 'w') as f:
                    json.dump(config, f, indent=2)
            except Exception as e:
                raise IOError(f"Failed to save config: {e}")

            # Atomic move from temp to final location
            for file in temp_dir.glob("*"):
                final_path = output_path / file.name
                if final_path.exists():
                    final_path.unlink()
                file.rename(final_path)

            logger.info(f"Models saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def load_model(self, model_dir: str = "models"):
        """Load pre-trained models with comprehensive validation."""
        model_path = Path(model_dir)

        # Validate directory exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        if not model_path.is_dir():
            raise ValueError(f"Path is not a directory: {model_dir}")

        # Check all required files exist
        required_files = [
            "content_model.pkl",
            "tfidf_vectorizer.pkl",
            "rating_scaler.pkl",
            "feature_matrix.pkl",
            "manhwa_catalog.pkl",
            "recommender_config.json"
        ]

        missing_files = [f for f in required_files if not (model_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required model files: {missing_files}")

        # Load and validate config first
        try:
            with open(model_path / "recommender_config.json", 'r') as f:
                config = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}")

        # Validate version compatibility
        model_version = config.get('version', '1.0.0')
        major_version = model_version.split('.')[0]
        current_major = MODEL_VERSION.split('.')[0]
        if major_version != current_major:
            raise ValueError(f"Incompatible model version: {model_version} vs {MODEL_VERSION}")
        if model_version != MODEL_VERSION:
            logger.warning(f"Model version mismatch: {model_version} vs {MODEL_VERSION}")

        # Load core components with error handling
        try:
            self.content_model = joblib.load(model_path / "content_model.pkl")
        except Exception as e:
            raise IOError(f"Failed to load content_model: {e}")

        try:
            self.tfidf_vectorizer = joblib.load(model_path / "tfidf_vectorizer.pkl")
        except Exception as e:
            raise IOError(f"Failed to load tfidf_vectorizer: {e}")

        try:
            self.rating_scaler = joblib.load(model_path / "rating_scaler.pkl")
        except Exception as e:
            raise IOError(f"Failed to load rating_scaler: {e}")

        try:
            self.feature_matrix = joblib.load(model_path / "feature_matrix.pkl")
        except Exception as e:
            raise IOError(f"Failed to load feature_matrix: {e}")

        # Load popularity scaler if it exists
        if (model_path / "popularity_scaler.pkl").exists():
            try:
                self.popularity_scaler = joblib.load(model_path / "popularity_scaler.pkl")
            except Exception as e:
                logger.warning(f"Failed to load popularity_scaler: {e}")

        # Load genre model (support both new and legacy naming)
        if (model_path / "genre_model.pkl").exists():
            try:
                self.genre_model = joblib.load(model_path / "genre_model.pkl")
                self.genre_features = joblib.load(model_path / "genre_features.pkl")
            except Exception as e:
                raise IOError(f"Failed to load genre_model: {e}")
        elif (model_path / "collab_model.pkl").exists():
            # Backward compatibility with old naming
            logger.info("Loading legacy collab_model (now genre_model)")
            try:
                self.genre_model = joblib.load(model_path / "collab_model.pkl")
                self.genre_features = joblib.load(model_path / "collab_features.pkl")
            except Exception as e:
                raise IOError(f"Failed to load legacy collab_model: {e}")

        try:
            self.df = pd.read_pickle(model_path / "manhwa_catalog.pkl")
        except Exception as e:
            raise IOError(f"Failed to load manhwa_catalog: {e}")

        # Load hyperparameters if available
        if 'hyperparameters' in config:
            self.hyperparameters = config['hyperparameters']
            self.weights = self.hyperparameters.get('weights', self.weights)
            self.tfidf_params = self.hyperparameters.get('tfidf', self.tfidf_params)
            self.svd_params = self.hyperparameters.get('svd', self.svd_params)
            self.knn_params = self.hyperparameters.get('knn', self.knn_params)
        else:
            # Load weights from config (legacy models)
            loaded_weights = config['weights']

            # Backward compatibility: convert old 'collaborative' key to 'genre_similarity'
            if 'collaborative' in loaded_weights and 'genre_similarity' not in loaded_weights:
                loaded_weights['genre_similarity'] = loaded_weights.pop('collaborative')
                logger.info("Converted legacy 'collaborative' weight to 'genre_similarity'")

            self.weights = loaded_weights

        logger.info(f"Models loaded from {model_path}")


def train_and_save_model(catalog_path: str, output_dir: str = "models", evaluate: bool = True, test_ratio: float = 0.2):
    """
    Train and save recommendation models with optional evaluation.

    Args:
        catalog_path: Path to manhwa catalog JSON file
        output_dir: Directory to save trained models
        evaluate: Whether to evaluate the model on a held-out test set
        test_ratio: Proportion of data to use for testing (only if evaluate=True)

    Returns:
        Tuple of (recommender, metrics) where metrics is None if evaluate=False
    """
    logger.info("Training hybrid recommendation model...")

    recommender = HybridManwhaRecommender()

    # Load data
    recommender.prepare_data(catalog_path)

    # Create train/test split if evaluating
    metrics = None
    if evaluate and len(recommender.df) > 20:  # Need enough data for meaningful split
        full_df = recommender.df.copy()
        train_df, test_df = recommender.create_evaluation_split(test_ratio=test_ratio)

        # Train on training set only
        recommender.df = train_df
        logger.info(f"Training on {len(train_df)} items, holding out {len(test_df)} for evaluation")

    # Build models
    recommender.build_content_features()
    recommender.build_genre_similarity_features()

    # Evaluate if requested
    if evaluate and len(recommender.df) > 20:
        metrics = recommender.evaluate_recommendations(test_df, k=10)

        # Retrain on full dataset for production
        logger.info("Retraining on full dataset for production...")
        recommender.df = full_df
        recommender.build_content_features()
        recommender.build_genre_similarity_features()

    # Save models
    recommender.save_model(output_dir)

    # Save evaluation metrics if available
    if metrics:
        import json
        metrics_path = Path(output_dir) / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation metrics saved to {metrics_path}")

    logger.info("Training complete!")

    return recommender, metrics


def main():
    """Test the hybrid recommender."""
    # Train on test data (use existing cleaned manwhas for now)
    recommender, metrics = train_and_save_model(
        catalog_path="data/cleanedManwhas.json",
        output_dir="models",
        evaluate=True
    )

    if metrics:
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Precision@10: {metrics['precision@k']:.4f}")
        print(f"Recall@10: {metrics['recall@k']:.4f}")
        print(f"NDCG@10: {metrics['ndcg@k']:.4f}")
        print(f"MRR: {metrics['mrr']:.4f}")
        print(f"Hit Rate@10: {metrics['hit_rate@k']:.4f}")
        print(f"Test items evaluated: {metrics['n_evaluated']}/{metrics['n_test_items']}")
        print("="*60 + "\n")

    # Test recommendations
    print("\n" + "="*60)
    print("TESTING HYBRID RECOMMENDER")
    print("="*60 + "\n")

    test_title = "Solo Leveling"
    print(f"Getting recommendations for: {test_title}\n")

    # Without user profile
    recs = recommender.recommend(test_title, n_recommendations=5)

    print("Top 5 Recommendations (No user profile):")
    for i, rec in enumerate(recs, 1):
        print(f"\n{i}. {rec['name']}")
        print(f"   Score: {rec['recommendation_score']:.3f}")
        print(f"   Rating: {rec['rating']}/5.0")
        print(f"   Genres: {', '.join(rec['genres'][:3])}")

    # With user profile
    print("\n" + "-"*60 + "\n")
    user_profile = {
        'liked_genres': ['Action', 'Fantasy'],
        'disliked_genres': ['Romance'],
        'min_rating': 4.0,
        'preferred_status': ['RELEASING', 'FINISHED']
    }

    print("User Profile:", json.dumps(user_profile, indent=2))

    recs_filtered = recommender.recommend(
        test_title,
        n_recommendations=5,
        user_profile=user_profile
    )

    print("\nTop 5 Recommendations (With user profile):")
    for i, rec in enumerate(recs_filtered, 1):
        print(f"\n{i}. {rec['name']}")
        print(f"   Score: {rec['recommendation_score']:.3f}")
        print(f"   Rating: {rec['rating']}/5.0")
        print(f"   Genres: {', '.join(rec['genres'][:3])}")
        print(f"   Status: {rec.get('status', 'Unknown')}")


if __name__ == "__main__":
    main()
