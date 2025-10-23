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
    - Content-based filtering (what it's about)
    - Collaborative filtering (what similar users like)
    - User preferences (what YOU like)
    """

    def __init__(self):
        self.df = None
        self.content_model = None
        self.collab_model = None
        self.tfidf_vectorizer = None
        self.rating_scaler = None
        self.feature_matrix = None
        self.user_preferences = {}

        # Weights for hybrid scoring
        self.weights = {
            'content': 0.4,      # Content similarity
            'collaborative': 0.3, # What similar users like
            'user_pref': 0.3     # Personal preferences
        }

    def prepare_data(self, catalog_path: str) -> pd.DataFrame:
        """Load and prepare manhwa catalog data."""
        logger.info(f"Loading catalog from {catalog_path}")

        with open(catalog_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.df = pd.DataFrame(data)

        # Handle missing values
        self.df['description'] = self.df['description'].fillna('')
        self.df['rating'] = self.df['rating'].fillna(self.df['rating'].median())

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

    def build_content_features(self):
        """Build content-based features using TF-IDF and metadata."""
        logger.info("Building content-based features...")

        # Combine text fields for TF-IDF
        text_features = []
        for idx, row in self.df.iterrows():
            # Combine description, genres, and tags
            text = row['description'] + ' '
            text += ' '.join(row['genres']) + ' '
            text += ' '.join(row['tags'])
            text_features.append(text)

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

        # Train KNN model
        self.content_model = NearestNeighbors(
            n_neighbors=min(21, len(self.df)),  # Top 20 + self
            metric='cosine',
            algorithm='brute'  # Better for sparse matrices
        )
        self.content_model.fit(self.feature_matrix)

        logger.info("Content-based model trained successfully")

    def build_collaborative_features(self, user_ratings_path: Optional[str] = None):
        """
        Build collaborative filtering model using matrix factorization.
        If no user ratings provided, use implicit feedback from popularity.
        """
        logger.info("Building collaborative filtering features...")

        if user_ratings_path and Path(user_ratings_path).exists():
            # Load actual user ratings
            with open(user_ratings_path, 'r') as f:
                user_ratings = json.load(f)

            # Build user-item matrix
            # TODO: Implement when we have user rating data
            logger.info("Using actual user ratings")
        else:
            # Use implicit feedback: create pseudo-users based on genre preferences
            logger.info("Using implicit feedback (genre-based)")

            # Create genre-based user profiles
            genre_profiles = self._create_genre_profiles()

            # Apply matrix factorization
            if len(genre_profiles) > 0:
                self.collab_model = TruncatedSVD(
                    n_components=min(50, len(genre_profiles) - 1),
                    random_state=42
                )
                self.collab_features = self.collab_model.fit_transform(genre_profiles)
                logger.info(f"Collaborative features shape: {self.collab_features.shape}")
            else:
                self.collab_features = None

    def _create_genre_profiles(self) -> np.ndarray:
        """Create genre-based profiles for collaborative filtering."""
        # Get all unique genres
        all_genres = set()
        for genres in self.df['genres']:
            all_genres.update(genres)

        all_genres = sorted(list(all_genres))

        # Create binary genre matrix
        genre_matrix = []
        for genres in self.df['genres']:
            profile = [1 if g in genres else 0 for g in all_genres]
            genre_matrix.append(profile)

        return np.array(genre_matrix)

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

    def get_collaborative_recommendations(
        self,
        manhwa_title: str,
        n_recommendations: int = 20
    ) -> List[Tuple[int, float]]:
        """Get collaborative filtering recommendations."""
        if self.collab_features is None:
            return []

        idx = self._find_manhwa_index(manhwa_title)
        if idx is None:
            return []

        # Compute similarity in latent space
        target_features = self.collab_features[idx]
        similarities = np.dot(self.collab_features, target_features)

        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]

        results = [
            (int(i), float(similarities[i]))
            for i in top_indices
        ]

        return results

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
        score = 0.5  # Base score

        manhwa = self.df.iloc[manhwa_idx]

        # Genre preferences
        liked_genres = set(user_profile.get('liked_genres', []))
        disliked_genres = set(user_profile.get('disliked_genres', []))
        manhwa_genres = set(manhwa['genres'])

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
        # Get recommendations from each method
        content_recs = self.get_content_recommendations(manhwa_title, n_recommendations=50)
        collab_recs = self.get_collaborative_recommendations(manhwa_title, n_recommendations=50)

        # Combine scores
        combined_scores = {}

        # Content-based scores
        for idx, score in content_recs:
            combined_scores[idx] = combined_scores.get(idx, 0) + self.weights['content'] * score

        # Collaborative scores
        for idx, score in collab_recs:
            combined_scores[idx] = combined_scores.get(idx, 0) + self.weights['collaborative'] * score

        # User preference scores
        if user_profile:
            for idx in combined_scores.keys():
                pref_score = self.get_user_preference_score(idx, user_profile)
                combined_scores[idx] += self.weights['user_pref'] * pref_score

        # Apply filters
        if filters:
            combined_scores = self._apply_filters(combined_scores, filters)

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
        """Find manhwa index by title with fuzzy matching."""
        from rapidfuzz import fuzz, process

        # Exact match first
        exact_match = self.df[self.df['name'].str.lower() == title.lower()]
        if not exact_match.empty:
            return exact_match.index[0]

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

    def save_model(self, output_dir: str = "models"):
        """Save trained models and data."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save models
        joblib.dump(self.content_model, output_path / "content_model.pkl")
        joblib.dump(self.tfidf_vectorizer, output_path / "tfidf_vectorizer.pkl")
        joblib.dump(self.rating_scaler, output_path / "rating_scaler.pkl")
        joblib.dump(self.feature_matrix, output_path / "feature_matrix.pkl")

        if self.collab_model:
            joblib.dump(self.collab_model, output_path / "collab_model.pkl")
            joblib.dump(self.collab_features, output_path / "collab_features.pkl")

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

    def load_model(self, model_dir: str = "models"):
        """Load pre-trained models."""
        model_path = Path(model_dir)

        self.content_model = joblib.load(model_path / "content_model.pkl")
        self.tfidf_vectorizer = joblib.load(model_path / "tfidf_vectorizer.pkl")
        self.rating_scaler = joblib.load(model_path / "rating_scaler.pkl")
        self.feature_matrix = joblib.load(model_path / "feature_matrix.pkl")

        if (model_path / "collab_model.pkl").exists():
            self.collab_model = joblib.load(model_path / "collab_model.pkl")
            self.collab_features = joblib.load(model_path / "collab_features.pkl")

        self.df = pd.read_pickle(model_path / "manhwa_catalog.pkl")

        with open(model_path / "recommender_config.json", 'r') as f:
            config = json.load(f)
            self.weights = config['weights']

        logger.info(f"Models loaded from {model_path}")


def train_and_save_model(catalog_path: str, output_dir: str = "models"):
    """Train and save recommendation models."""
    logger.info("Training hybrid recommendation model...")

    recommender = HybridManwhaRecommender()

    # Load data
    recommender.prepare_data(catalog_path)

    # Build models
    recommender.build_content_features()
    recommender.build_collaborative_features()

    # Save models
    recommender.save_model(output_dir)

    logger.info("Training complete!")

    return recommender


def main():
    """Test the hybrid recommender."""
    # Train on test data (use existing cleaned manwhas for now)
    recommender = train_and_save_model(
        catalog_path="data/cleanedManwhas.json",
        output_dir="models"
    )

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
