"""
Advanced Hybrid Recommendation Engine combining multiple approaches:
1. Content-Based Filtering (TF-IDF + KNN)
2. Collaborative Filtering (Matrix Factorization)
3. User Preference Learning
4. Demographic Filtering
"""

# REVIEW: [MEDIUM] Logging configuration at module level
# Recommendation: Move basicConfig to application entry point to avoid conflicts
# Location: Lines 22-23
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix
import joblib
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def __init__(self):
        self.df = None
        self.content_model = None
        self.genre_model = None  # Renamed from collab_model - this is genre-based similarity, not collaborative filtering
        self.tfidf_vectorizer = None
        self.rating_scaler = None
        self.feature_matrix = None
        self.user_preferences = {}

        # REVIEW: [HIGH] Hardcoded weights without justification
        # Recommendation: Add hyperparameter tuning (grid search/Bayesian optimization) to find optimal weights
        # Consider exposing these as constructor parameters for easier experimentation
        # Weights for hybrid scoring
        self.weights = {
            'content': 0.4,          # Content similarity (TF-IDF)
            'genre_similarity': 0.3, # Genre-based similarity (renamed from 'collaborative')
            'user_pref': 0.3         # Personal preferences
        }

    # REVIEW: [HIGH] No input validation or error handling for missing/corrupt files
    # Recommendation: Add try-except with specific error messages, validate file exists and is valid JSON
    # Location: prepare_data function
    def prepare_data(self, catalog_path: str) -> pd.DataFrame:
        """Load and prepare manhwa catalog data."""
        # REVIEW: [CRITICAL] No validation of minimum data requirements
        # Recommendation: Check for minimum number of entries, required columns before proceeding
        # Location: Lines 50-94
        logger.info(f"Loading catalog from {catalog_path}")

        with open(catalog_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.df = pd.DataFrame(data)

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
        Evaluate recommendation quality using standard metrics.

        For each manhwa in test set, generate recommendations and measure:
        - Precision@K: How many of top K recs share genres with the test item
        - Recall@K: Of all items that share genres, how many are in top K
        - NDCG@K: Normalized Discounted Cumulative Gain
        - MRR: Mean Reciprocal Rank
        - Hit Rate@K: Percentage where at least 1 relevant item in top K

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

        # Calculate final metrics
        metrics = {
            'precision@k': np.mean(precision_scores) if precision_scores else 0.0,
            'recall@k': np.mean(recall_scores) if recall_scores else 0.0,
            'ndcg@k': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'hit_rate@k': hits / len(test_df) if len(test_df) > 0 else 0.0,
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

        # REVIEW: [HIGH] TF-IDF hyperparameters are not justified
        # Recommendation: Add grid search to tune max_features, min_df, max_df
        # Consider removing 'english' stop_words for non-English manhwa titles/descriptions
        # Location: Lines 110-116
        # TF-IDF vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,
            max_df=0.8
        )

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Normalize ratings
        self.rating_scaler = MinMaxScaler()
        scaled_ratings = self.rating_scaler.fit_transform(
            self.df[['rating']].values
        )

        # REVIEW: [MEDIUM] Popularity scaler is not saved/reused
        # Recommendation: Store popularity_scaler as instance variable for consistency in prediction
        # Location: Lines 128-131
        # Normalize popularity
        popularity_scaler = MinMaxScaler()
        scaled_popularity = popularity_scaler.fit_transform(
            self.df[['popularity']].values
        )

        # Combine features
        self.feature_matrix = hstack([
            tfidf_matrix,
            csr_matrix(scaled_ratings),
            csr_matrix(scaled_popularity)
        ])

        logger.info(f"Combined feature matrix shape: {self.feature_matrix.shape}")

        # REVIEW: [MEDIUM] Using cosine distance but features aren't L2-normalized
        # Recommendation: Normalize feature_matrix rows to unit length for true cosine similarity
        # Or use normalized vectors: from sklearn.preprocessing import normalize
        # Location: Lines 143-148
        # Train KNN model
        self.content_model = NearestNeighbors(
            n_neighbors=min(21, len(self.df)),  # Top 20 + self
            metric='cosine',
            algorithm='brute'  # Better for sparse matrices
        )
        self.content_model.fit(self.feature_matrix)

        logger.info("Content-based model trained successfully")

    def build_genre_similarity_features(self):
        """
        Build genre-based similarity features using SVD on genre co-occurrence matrix.

        This creates a lower-dimensional representation of manhwa based on their genres,
        allowing us to find similar items based on genre patterns. This is content-based
        filtering, not collaborative filtering (which would require user interaction data).
        """
        logger.info("Building genre-based similarity features...")

        # Create genre-based profiles (binary matrix of genre presence)
        genre_profiles = self._create_genre_profiles()

        # REVIEW: [MEDIUM] SVD n_components=50 is arbitrary
        # Recommendation: Use explained_variance_ratio_ to choose optimal components or tune via CV
        # Apply dimensionality reduction to capture genre patterns
        if len(genre_profiles) > 0:
            self.genre_model = TruncatedSVD(
                n_components=min(50, len(genre_profiles) - 1),
                random_state=42
            )
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

    # REVIEW: [MEDIUM] No validation that model has been trained
    # Recommendation: Add check if self.content_model is None, raise informative error
    # Location: get_content_recommendations function
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

    # REVIEW: [HIGH] No error handling or logging when input manhwa not found
    # Recommendation: Log warning with suggestions for similar titles
    # Location: recommend function
    def recommend(
        self,
        manhwa_title: str,
        n_recommendations: int = 10,
        user_profile: Optional[Dict] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get hybrid recommendations combining all methods.

        Args:
            manhwa_title: Title of manhwa to base recommendations on
            n_recommendations: Number of recommendations to return
            user_profile: User preference profile
            filters: Additional filters (genre, rating, status, etc.)

        Returns:
            List of recommended manhwa with scores
        """
        # REVIEW: [MEDIUM] Silent failure returns empty list - should log error
        # Location: Lines 326-328
        # Find input manhwa index to exclude it from results
        input_idx = self._find_manhwa_index(manhwa_title)
        if input_idx is None:
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

        # REVIEW: [MEDIUM] Weights don't sum to 1.0 when user_profile is None
        # Recommendation: Normalize weights dynamically based on which components are active
        # Location: Lines 351-354
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

    # REVIEW: [MEDIUM] Fuzzy matching computed fresh every time - no caching
    # Recommendation: Cache fuzzy match results or pre-build index for common queries
    # Location: _find_manhwa_index function
    def _find_manhwa_index(self, title: str) -> Optional[int]:
        """Find manhwa index by title with fuzzy matching."""
        # REVIEW: [LOW] Import inside function - move to top of file
        # Location: Line 418
        from rapidfuzz import fuzz, process

        # Exact match first
        exact_match = self.df[self.df['name'].str.lower() == title.lower()]
        if not exact_match.empty:
            return exact_match.index[0]

        # REVIEW: [MEDIUM] Converting entire column to list on every call is wasteful
        # Recommendation: Cache the titles list as instance variable
        # Location: Line 426
        # Fuzzy match
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

    # REVIEW: [MEDIUM] No error handling for file I/O operations
    # Recommendation: Wrap in try-except, validate write permissions before starting
    # Location: save_model function
    def save_model(self, output_dir: str = "models"):
        """Save trained models and data."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save models
        joblib.dump(self.content_model, output_path / "content_model.pkl")
        joblib.dump(self.tfidf_vectorizer, output_path / "tfidf_vectorizer.pkl")
        joblib.dump(self.rating_scaler, output_path / "rating_scaler.pkl")
        joblib.dump(self.feature_matrix, output_path / "feature_matrix.pkl")

        if self.genre_model:
            joblib.dump(self.genre_model, output_path / "genre_model.pkl")
            joblib.dump(self.genre_features, output_path / "genre_features.pkl")

        # Save dataframe
        self.df.to_pickle(output_path / "manhwa_catalog.pkl")

        # Save config
        config = {
            'weights': self.weights,
            'n_entries': len(self.df)
        }
        with open(output_path / "recommender_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Models saved to {output_path}")

    # REVIEW: [HIGH] No validation that files exist before loading
    # Recommendation: Check file existence, add try-except with informative errors
    # Location: load_model function
    def load_model(self, model_dir: str = "models"):
        """Load pre-trained models."""
        model_path = Path(model_dir)

        # REVIEW: [MEDIUM] No version checking - models could be incompatible
        # Recommendation: Save and validate model version/schema
        # Location: Lines 474-489
        self.content_model = joblib.load(model_path / "content_model.pkl")
        self.tfidf_vectorizer = joblib.load(model_path / "tfidf_vectorizer.pkl")
        self.rating_scaler = joblib.load(model_path / "rating_scaler.pkl")
        self.feature_matrix = joblib.load(model_path / "feature_matrix.pkl")

        # Load genre model (support both new and legacy naming)
        if (model_path / "genre_model.pkl").exists():
            self.genre_model = joblib.load(model_path / "genre_model.pkl")
            self.genre_features = joblib.load(model_path / "genre_features.pkl")
        elif (model_path / "collab_model.pkl").exists():
            # Backward compatibility with old naming
            logger.info("Loading legacy collab_model (now genre_model)")
            self.genre_model = joblib.load(model_path / "collab_model.pkl")
            self.genre_features = joblib.load(model_path / "collab_features.pkl")

        self.df = pd.read_pickle(model_path / "manhwa_catalog.pkl")

        with open(model_path / "recommender_config.json", 'r') as f:
            config = json.load(f)
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
