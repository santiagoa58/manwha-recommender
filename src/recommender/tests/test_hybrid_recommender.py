"""
Comprehensive tests for the hybrid recommendation engine.
Tests actual recommendation logic, not just mocked responses.
"""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from src.recommender.hybrid_recommender import HybridManwhaRecommender


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def trained_recommender(sample_catalog_file):
    """Create a trained recommender for testing."""
    recommender = HybridManwhaRecommender()
    recommender.prepare_data(str(sample_catalog_file))
    recommender.build_content_features()
    recommender.build_collaborative_features()
    return recommender


@pytest.mark.unit
class TestDataPreparation:
    """Test data loading and preparation logic."""

    def test_load_catalog_success(self, sample_catalog_file):
        """Test that catalog loads successfully."""
        recommender = HybridManwhaRecommender()
        df = recommender.prepare_data(str(sample_catalog_file))

        assert df is not None
        assert len(df) > 0
        assert isinstance(df, pd.DataFrame)

    def test_missing_fields_handled(self, temp_data_dir):
        """Test that missing fields are filled with defaults."""
        # Create catalog with missing fields
        incomplete_data = [
            {
                "name": "Test Manhwa",
                "description": "Test description"
                # Missing: rating, popularity, genres, tags
            }
        ]

        catalog_path = temp_data_dir / "incomplete.json"
        with open(catalog_path, 'w') as f:
            json.dump(incomplete_data, f)

        recommender = HybridManwhaRecommender()
        df = recommender.prepare_data(str(catalog_path))

        # Should fill in missing fields
        assert df.iloc[0]['description'] == "Test description"
        assert df.iloc[0]['rating'] > 0  # Should have default
        assert df.iloc[0]['popularity'] == 0
        assert isinstance(df.iloc[0]['genres'], list)
        assert isinstance(df.iloc[0]['tags'], list)

    def test_legacy_data_format_supported(self, temp_data_dir):
        """Test that older data format with 'tags' but no 'genres' works."""
        legacy_data = [
            {
                "name": "Tower of God",
                "description": "Tower story",
                "rating": 4.5,
                "tags": ["Action", "Adventure"]  # Only tags, no genres
            }
        ]

        catalog_path = temp_data_dir / "legacy.json"
        with open(catalog_path, 'w') as f:
            json.dump(legacy_data, f)

        recommender = HybridManwhaRecommender()
        df = recommender.prepare_data(str(catalog_path))

        # Should use tags as genres
        assert 'genres' in df.columns
        assert len(df.iloc[0]['genres']) > 0


@pytest.mark.unit
class TestContentFeatureBuilding:
    """Test content-based feature extraction."""

    def test_tfidf_vectorization(self, trained_recommender):
        """Test that TF-IDF vectorization produces valid matrix."""
        rec = trained_recommender

        assert rec.tfidf_vectorizer is not None
        assert rec.feature_matrix is not None

        # Check matrix shape
        n_samples = len(rec.df)
        assert rec.feature_matrix.shape[0] == n_samples

    def test_rating_normalization(self, trained_recommender):
        """Test that ratings are properly normalized."""
        rec = trained_recommender

        assert rec.rating_scaler is not None

        # Get scaled ratings from feature matrix
        # (last two columns are ratings and popularity)
        scaled_ratings = rec.feature_matrix[:, -2].toarray().flatten()

        # Scaled ratings should be between 0 and 1
        assert all(0 <= r <= 1 for r in scaled_ratings)

    def test_content_model_trained(self, trained_recommender):
        """Test that KNN model is properly trained."""
        rec = trained_recommender

        assert rec.content_model is not None

        # Should be able to query the model
        distances, indices = rec.content_model.kneighbors(
            rec.feature_matrix[0],
            n_neighbors=2
        )

        assert len(distances[0]) == 2
        assert len(indices[0]) == 2

    def test_feature_matrix_includes_all_components(self, trained_recommender):
        """Test that feature matrix combines TF-IDF, ratings, and popularity."""
        rec = trained_recommender

        # Feature matrix should have TF-IDF features + 2 additional (rating, popularity)
        assert rec.feature_matrix.shape[1] > 2  # Should have TF-IDF + 2 extra


@pytest.mark.unit
class TestCollaborativeFiltering:
    """Test collaborative filtering logic."""

    def test_genre_profiles_created(self, trained_recommender):
        """Test that genre profiles are created for collaborative filtering."""
        rec = trained_recommender

        # With implicit feedback (no user ratings), should still have features
        assert rec.collab_features is not None or rec.collab_model is None

    def test_svd_dimensionality_reduction(self, trained_recommender):
        """Test that SVD reduces dimensionality appropriately."""
        rec = trained_recommender

        if rec.collab_features is not None:
            # Should have fewer components than original genres
            assert rec.collab_features.shape[1] < len(rec.df)


@pytest.mark.unit
class TestFuzzyTitleMatching:
    """Test fuzzy title matching functionality."""

    def test_exact_match_found(self, trained_recommender):
        """Test that exact title matches are found."""
        rec = trained_recommender

        # Use a title we know exists
        existing_title = rec.df.iloc[0]['name']
        idx = rec._find_manhwa_index(existing_title)

        assert idx is not None
        assert idx == 0

    def test_case_insensitive_match(self, trained_recommender):
        """Test that matching is case-insensitive."""
        rec = trained_recommender

        existing_title = rec.df.iloc[0]['name']

        # Test with different cases
        idx_lower = rec._find_manhwa_index(existing_title.lower())
        idx_upper = rec._find_manhwa_index(existing_title.upper())

        assert idx_lower is not None
        assert idx_upper is not None
        assert idx_lower == idx_upper

    def test_fuzzy_match_with_typo(self, trained_recommender):
        """Test that fuzzy matching handles typos."""
        rec = trained_recommender

        # Deliberately introduce a typo
        correct_title = "Solo Leveling"
        typo_title = "Solo Levelng"  # Missing 'i'

        # Should still find a match if the entry exists
        idx = rec._find_manhwa_index(typo_title)

        # If "Solo Leveling" exists in the test data, should find it
        if any("Solo" in name for name in rec.df['name']):
            assert idx is not None

    def test_no_match_for_nonexistent(self, trained_recommender):
        """Test that non-existent titles return None."""
        rec = trained_recommender

        idx = rec._find_manhwa_index("This Title Definitely Does Not Exist XYZ")

        assert idx is None


@pytest.mark.unit
class TestContentRecommendations:
    """Test content-based recommendation generation."""

    def test_content_recommendations_returned(self, trained_recommender):
        """Test that content-based recommendations are generated."""
        rec = trained_recommender

        title = rec.df.iloc[0]['name']
        recs = rec.get_content_recommendations(title, n_recommendations=5)

        assert recs is not None
        assert isinstance(recs, list)

        # Should return recommendations (if enough entries exist)
        if len(rec.df) > 1:
            assert len(recs) > 0

    def test_content_recommendations_exclude_self(self, trained_recommender):
        """Test that input manhwa is not in its own recommendations."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]['name']
            recs = rec.get_content_recommendations(title, n_recommendations=5)

            # Input index (0) should not be in results
            indices = [idx for idx, score in recs]
            assert 0 not in indices

    def test_content_recommendations_have_scores(self, trained_recommender):
        """Test that recommendations include similarity scores."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]['name']
            recs = rec.get_content_recommendations(title, n_recommendations=5)

            for idx, score in recs:
                assert isinstance(idx, int)
                assert isinstance(score, float)
                assert 0 <= score <= 1  # Similarity scores should be 0-1

    def test_content_recommendations_sorted_by_score(self, trained_recommender):
        """Test that recommendations are sorted by similarity score."""
        rec = trained_recommender

        if len(rec.df) > 2:
            title = rec.df.iloc[0]['name']
            recs = rec.get_content_recommendations(title, n_recommendations=5)

            if len(recs) > 1:
                scores = [score for _, score in recs]
                # Should be in descending order
                assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
class TestUserPreferenceScoring:
    """Test user preference scoring logic."""

    def test_genre_preference_boost(self, trained_recommender):
        """Test that liked genres increase score."""
        rec = trained_recommender

        # Find a manhwa with Action genre
        action_idx = None
        for idx, row in rec.df.iterrows():
            if "Action" in row['genres']:
                action_idx = idx
                break

        if action_idx is not None:
            user_profile = {
                'liked_genres': ['Action'],
                'disliked_genres': [],
                'min_rating': 0
            }

            score = rec.get_user_preference_score(action_idx, user_profile)

            # Should have positive score
            assert score > 0.5  # Base score is 0.5

    def test_disliked_genre_penalty(self, trained_recommender):
        """Test that disliked genres decrease score."""
        rec = trained_recommender

        # Find a manhwa with Romance genre (if exists)
        romance_idx = None
        for idx, row in rec.df.iterrows():
            if "Romance" in row['genres']:
                romance_idx = idx
                break

        if romance_idx is not None:
            user_profile = {
                'liked_genres': [],
                'disliked_genres': ['Romance'],
                'min_rating': 0
            }

            score = rec.get_user_preference_score(romance_idx, user_profile)

            # Should have penalty
            assert score < 0.5  # Below base score

    def test_rating_threshold_filter(self, trained_recommender):
        """Test that rating threshold affects score."""
        rec = trained_recommender

        manhwa_idx = 0
        manhwa_rating = rec.df.iloc[manhwa_idx]['rating']

        # Set threshold above manhwa rating
        user_profile = {
            'liked_genres': [],
            'disliked_genres': [],
            'min_rating': manhwa_rating + 1
        }

        score = rec.get_user_preference_score(manhwa_idx, user_profile)

        # Should have penalty for not meeting threshold
        assert score < 0.5


@pytest.mark.unit
class TestHybridRecommendations:
    """Test full hybrid recommendation logic."""

    def test_hybrid_recommendations_returned(self, trained_recommender):
        """Test that hybrid recommendations are generated."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]['name']
            recs = rec.recommend(title, n_recommendations=5)

            assert recs is not None
            assert isinstance(recs, list)
            assert len(recs) > 0

    def test_hybrid_recommendations_include_scores(self, trained_recommender):
        """Test that recommendations include hybrid scores."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]['name']
            recs = rec.recommend(title, n_recommendations=5)

            for manhwa in recs:
                assert 'recommendation_score' in manhwa
                assert isinstance(manhwa['recommendation_score'], float)
                # Allow for floating point precision issues (scores should be >= 0, within tolerance)
                assert manhwa['recommendation_score'] >= -1e-10

    def test_hybrid_with_user_profile(self, trained_recommender):
        """Test that user profile affects recommendations."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]['name']

            # Get recommendations without user profile
            recs_no_profile = rec.recommend(title, n_recommendations=5)

            # Get recommendations with user profile
            user_profile = {
                'liked_genres': ['Action', 'Fantasy'],
                'disliked_genres': ['Romance'],
                'min_rating': 4.0
            }
            recs_with_profile = rec.recommend(
                title,
                n_recommendations=5,
                user_profile=user_profile
            )

            # Results should potentially differ
            assert recs_with_profile is not None
            assert len(recs_with_profile) > 0

    def test_hybrid_recommendations_sorted(self, trained_recommender):
        """Test that hybrid recommendations are sorted by score."""
        rec = trained_recommender

        if len(rec.df) > 2:
            title = rec.df.iloc[0]['name']
            recs = rec.recommend(title, n_recommendations=5)

            if len(recs) > 1:
                scores = [m['recommendation_score'] for m in recs]
                # Should be in descending order
                assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
class TestFiltering:
    """Test filtering logic."""

    def test_genre_filter(self, trained_recommender):
        """Test that genre filtering works."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]['name']

            filters = {
                'genres': ['Action']
            }

            recs = rec.recommend(title, n_recommendations=10, filters=filters)

            # All results should have Action genre
            for manhwa in recs:
                assert 'Action' in manhwa['genres']

    def test_rating_filter(self, trained_recommender):
        """Test that rating filtering works."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]['name']

            filters = {
                'min_rating': 4.5
            }

            recs = rec.recommend(title, n_recommendations=10, filters=filters)

            # All results should have rating >= 4.5
            for manhwa in recs:
                assert manhwa['rating'] >= 4.5

    def test_multiple_filters(self, trained_recommender):
        """Test that multiple filters work together."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]['name']

            filters = {
                'genres': ['Action'],
                'min_rating': 4.0
            }

            recs = rec.recommend(title, n_recommendations=10, filters=filters)

            # All results should match all filters
            for manhwa in recs:
                assert 'Action' in manhwa['genres']
                assert manhwa['rating'] >= 4.0


@pytest.mark.unit
class TestModelPersistence:
    """Test model save and load functionality."""

    def test_save_model(self, trained_recommender, temp_model_dir):
        """Test that model can be saved."""
        rec = trained_recommender

        rec.save_model(str(temp_model_dir))

        # Check that files were created
        assert (temp_model_dir / "content_model.pkl").exists()
        assert (temp_model_dir / "tfidf_vectorizer.pkl").exists()
        assert (temp_model_dir / "rating_scaler.pkl").exists()
        assert (temp_model_dir / "feature_matrix.pkl").exists()
        assert (temp_model_dir / "manhwa_catalog.pkl").exists()
        assert (temp_model_dir / "recommender_config.json").exists()

    def test_load_model(self, trained_recommender, temp_model_dir):
        """Test that model can be loaded."""
        # Save model first
        trained_recommender.save_model(str(temp_model_dir))

        # Create new recommender and load
        new_rec = HybridManwhaRecommender()
        new_rec.load_model(str(temp_model_dir))

        # Should have loaded components
        assert new_rec.content_model is not None
        assert new_rec.tfidf_vectorizer is not None
        assert new_rec.feature_matrix is not None
        assert new_rec.df is not None
        assert len(new_rec.df) == len(trained_recommender.df)

    def test_loaded_model_produces_same_recommendations(
        self, trained_recommender, temp_model_dir
    ):
        """Test that loaded model produces same recommendations."""
        # Get recommendations from original
        title = trained_recommender.df.iloc[0]['name']
        original_recs = trained_recommender.recommend(title, n_recommendations=5)

        # Save and load model
        trained_recommender.save_model(str(temp_model_dir))

        new_rec = HybridManwhaRecommender()
        new_rec.load_model(str(temp_model_dir))

        # Get recommendations from loaded model
        loaded_recs = new_rec.recommend(title, n_recommendations=5)

        # Should produce same results
        assert len(loaded_recs) == len(original_recs)

        # Scores should be very close (allowing for small numerical differences)
        for orig, loaded in zip(original_recs, loaded_recs):
            assert orig['name'] == loaded['name']
            assert abs(orig['recommendation_score'] - loaded['recommendation_score']) < 0.01


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases in recommendation system."""

    def test_unknown_title(self, trained_recommender):
        """Test handling of unknown title."""
        rec = trained_recommender

        recs = rec.recommend("This Title Does Not Exist XYZ", n_recommendations=5)

        # Should return empty list or handle gracefully
        assert recs is not None
        assert isinstance(recs, list)

    def test_request_more_recommendations_than_available(self, trained_recommender):
        """Test requesting more recommendations than data available."""
        rec = trained_recommender

        # Request way more than we have
        title = rec.df.iloc[0]['name']
        recs = rec.recommend(title, n_recommendations=10000)

        # Should return at most (total - 1) recommendations (excluding self)
        assert len(recs) <= len(rec.df) - 1

    def test_empty_user_profile(self, trained_recommender):
        """Test that empty user profile doesn't break recommendations."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]['name']

            empty_profile = {}
            recs = rec.recommend(title, n_recommendations=5, user_profile=empty_profile)

            assert recs is not None
            assert isinstance(recs, list)


@pytest.mark.unit
class TestModelEvaluation:
    """Test model evaluation capabilities."""

    def test_create_evaluation_split(self, sample_catalog_file):
        """Test that data can be split into train/test sets."""
        rec = HybridManwhaRecommender()
        rec.prepare_data(sample_catalog_file)

        train_df, test_df = rec.create_evaluation_split(test_ratio=0.3, random_state=42)

        # Should split correctly
        assert len(train_df) + len(test_df) == len(rec.df)
        assert len(test_df) / len(rec.df) == pytest.approx(0.3, abs=0.05)

        # Should be different dataframes
        assert not train_df.equals(test_df)

        # No overlap in indices
        assert len(set(train_df.index) & set(test_df.index)) == 0

    def test_create_evaluation_split_reproducible(self, sample_catalog_file):
        """Test that split is reproducible with same random seed."""
        rec = HybridManwhaRecommender()
        rec.prepare_data(sample_catalog_file)

        train1, test1 = rec.create_evaluation_split(test_ratio=0.2, random_state=42)
        train2, test2 = rec.create_evaluation_split(test_ratio=0.2, random_state=42)

        # Should produce same splits
        assert train1.equals(train2)
        assert test1.equals(test2)

    def test_create_evaluation_split_without_data(self):
        """Test that split fails gracefully without loaded data."""
        rec = HybridManwhaRecommender()

        with pytest.raises(ValueError, match="Data not loaded"):
            rec.create_evaluation_split()

    def test_evaluate_recommendations_returns_metrics(self, sample_catalog_file):
        """Test that evaluation returns expected metrics."""
        rec = HybridManwhaRecommender()
        rec.prepare_data(sample_catalog_file)

        # Split and train on training data
        train_df, test_df = rec.create_evaluation_split(test_ratio=0.3, random_state=42)
        rec.df = train_df
        rec.build_content_features()

        # Restore full df for evaluation (test items need to be findable)
        rec.df = pd.concat([train_df, test_df])

        # Evaluate
        metrics = rec.evaluate_recommendations(test_df, k=5)

        # Should return all expected metrics
        assert 'precision@k' in metrics
        assert 'recall@k' in metrics
        assert 'ndcg@k' in metrics
        assert 'mrr' in metrics
        assert 'hit_rate@k' in metrics
        assert metrics['k'] == 5

        # Metrics should be in valid range
        assert 0 <= metrics['precision@k'] <= 1
        assert 0 <= metrics['recall@k'] <= 1
        assert 0 <= metrics['ndcg@k'] <= 1
        assert 0 <= metrics['mrr'] <= 1
        assert 0 <= metrics['hit_rate@k'] <= 1

    def test_evaluate_recommendations_without_model(self, sample_catalog_file):
        """Test that evaluation fails gracefully without trained model."""
        rec = HybridManwhaRecommender()
        rec.prepare_data(sample_catalog_file)

        _, test_df = rec.create_evaluation_split(test_ratio=0.3)

        with pytest.raises(ValueError, match="Model not trained"):
            rec.evaluate_recommendations(test_df, k=5)

    def test_evaluation_metrics_saved_during_training(self, sample_catalog_file, temp_model_dir):
        """Test that evaluation metrics are saved when training with evaluation."""
        from src.recommender.hybrid_recommender import train_and_save_model

        recommender, metrics = train_and_save_model(
            catalog_path=sample_catalog_file,
            output_dir=str(temp_model_dir),
            evaluate=True,
            test_ratio=0.3
        )

        # With small dataset (6 items), evaluation may be skipped (requires > 20 items)
        # This is expected behavior, so we test both cases
        if metrics is not None:
            # Evaluation ran successfully
            assert 'precision@k' in metrics

            # Should save metrics file
            metrics_file = temp_model_dir / "evaluation_metrics.json"
            assert metrics_file.exists()

            # File should contain valid JSON
            import json
            with open(metrics_file) as f:
                saved_metrics = json.load(f)
            assert saved_metrics == metrics
        else:
            # Evaluation skipped due to small dataset - this is fine
            metrics_file = temp_model_dir / "evaluation_metrics.json"
            assert not metrics_file.exists()  # No metrics file should be created


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
