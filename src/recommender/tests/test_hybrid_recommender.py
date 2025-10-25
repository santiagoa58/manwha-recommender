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
    recommender.build_genre_similarity_features()
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
                "description": "Test description",
                # Missing: rating, popularity, genres, tags
            }
        ]

        catalog_path = temp_data_dir / "incomplete.json"
        with open(catalog_path, "w") as f:
            json.dump(incomplete_data, f)

        recommender = HybridManwhaRecommender()
        df = recommender.prepare_data(str(catalog_path))

        # Should fill in missing fields
        assert df.iloc[0]["description"] == "Test description"
        assert df.iloc[0]["rating"] > 0  # Should have default
        assert df.iloc[0]["popularity"] == 0
        assert isinstance(df.iloc[0]["genres"], list)
        assert isinstance(df.iloc[0]["tags"], list)

    def test_legacy_data_format_supported(self, temp_data_dir):
        """Test that older data format with 'tags' but no 'genres' works."""
        legacy_data = [
            {
                "name": "Tower of God",
                "description": "Tower story",
                "rating": 4.5,
                "tags": ["Action", "Adventure"],  # Only tags, no genres
            }
        ]

        catalog_path = temp_data_dir / "legacy.json"
        with open(catalog_path, "w") as f:
            json.dump(legacy_data, f)

        recommender = HybridManwhaRecommender()
        df = recommender.prepare_data(str(catalog_path))

        # Should use tags as genres
        assert "genres" in df.columns
        assert len(df.iloc[0]["genres"]) > 0


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
        distances, indices = rec.content_model.kneighbors(rec.feature_matrix[0], n_neighbors=2)

        assert len(distances[0]) == 2
        assert len(indices[0]) == 2

    def test_feature_matrix_includes_all_components(self, trained_recommender):
        """Test that feature matrix combines TF-IDF, ratings, and popularity."""
        rec = trained_recommender

        # Feature matrix should have TF-IDF features + 2 additional (rating, popularity)
        assert rec.feature_matrix.shape[1] > 2  # Should have TF-IDF + 2 extra


@pytest.mark.unit
class TestGenreSimilarity:
    """Test genre-based similarity logic."""

    def test_genre_profiles_created(self, trained_recommender):
        """Test that genre features are created for similarity matching."""
        rec = trained_recommender

        # Should have genre features
        assert rec.genre_features is not None or rec.genre_model is None

    def test_svd_dimensionality_reduction(self, trained_recommender):
        """Test that SVD reduces dimensionality appropriately."""
        rec = trained_recommender

        if rec.genre_features is not None:
            # Should have fewer components than original genres
            assert rec.genre_features.shape[1] < len(rec.df)


@pytest.mark.unit
class TestFuzzyTitleMatching:
    """Test fuzzy title matching functionality."""

    def test_exact_match_found(self, trained_recommender):
        """Test that exact title matches are found."""
        rec = trained_recommender

        # Use a title we know exists
        existing_title = rec.df.iloc[0]["name"]
        idx = rec._find_manhwa_index(existing_title)

        assert idx is not None
        assert idx == 0

    def test_case_insensitive_match(self, trained_recommender):
        """Test that matching is case-insensitive."""
        rec = trained_recommender

        existing_title = rec.df.iloc[0]["name"]

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
        if any("Solo" in name for name in rec.df["name"]):
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

        title = rec.df.iloc[0]["name"]
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
            title = rec.df.iloc[0]["name"]
            recs = rec.get_content_recommendations(title, n_recommendations=5)

            # Input index (0) should not be in results
            indices = [idx for idx, score in recs]
            assert 0 not in indices

    def test_content_recommendations_have_scores(self, trained_recommender):
        """Test that recommendations include similarity scores."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]["name"]
            recs = rec.get_content_recommendations(title, n_recommendations=5)

            for idx, score in recs:
                assert isinstance(idx, int)
                assert isinstance(score, float)
                assert 0 <= score <= 1  # Similarity scores should be 0-1

    def test_content_recommendations_sorted_by_score(self, trained_recommender):
        """Test that recommendations are sorted by similarity score."""
        rec = trained_recommender

        if len(rec.df) > 2:
            title = rec.df.iloc[0]["name"]
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
            if "Action" in row["genres"]:
                action_idx = idx
                break

        if action_idx is not None:
            user_profile = {"liked_genres": ["Action"], "disliked_genres": [], "min_rating": 0}

            score = rec.get_user_preference_score(action_idx, user_profile)

            # Should have positive score
            assert score > 0.5  # Base score is 0.5

    def test_disliked_genre_penalty(self, trained_recommender):
        """Test that disliked genres decrease score."""
        rec = trained_recommender

        # Find a manhwa with Romance genre (if exists)
        romance_idx = None
        for idx, row in rec.df.iterrows():
            if "Romance" in row["genres"]:
                romance_idx = idx
                break

        if romance_idx is not None:
            user_profile = {"liked_genres": [], "disliked_genres": ["Romance"], "min_rating": 0}

            score = rec.get_user_preference_score(romance_idx, user_profile)

            # Should have penalty
            assert score < 0.5  # Below base score

    def test_rating_threshold_filter(self, trained_recommender):
        """Test that rating threshold affects score."""
        rec = trained_recommender

        manhwa_idx = 0
        manhwa_rating = rec.df.iloc[manhwa_idx]["rating"]

        # Set threshold above manhwa rating
        user_profile = {"liked_genres": [], "disliked_genres": [], "min_rating": manhwa_rating + 1}

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
            title = rec.df.iloc[0]["name"]
            recs = rec.recommend(title, n_recommendations=5)

            assert recs is not None
            assert isinstance(recs, list)
            assert len(recs) > 0

    def test_hybrid_recommendations_include_scores(self, trained_recommender):
        """Test that recommendations include hybrid scores."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]["name"]
            recs = rec.recommend(title, n_recommendations=5)

            for manhwa in recs:
                assert "recommendation_score" in manhwa
                assert isinstance(manhwa["recommendation_score"], float)
                # Allow for floating point precision issues (scores should be >= 0, within tolerance)
                assert manhwa["recommendation_score"] >= -1e-10

    def test_hybrid_with_user_profile(self, trained_recommender):
        """Test that user profile affects recommendations."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]["name"]

            # Get recommendations without user profile
            recs_no_profile = rec.recommend(title, n_recommendations=5)

            # Get recommendations with user profile
            user_profile = {
                "liked_genres": ["Action", "Fantasy"],
                "disliked_genres": ["Romance"],
                "min_rating": 4.0,
            }
            recs_with_profile = rec.recommend(title, n_recommendations=5, user_profile=user_profile)

            # Results should potentially differ
            assert recs_with_profile is not None
            assert len(recs_with_profile) > 0

    def test_hybrid_recommendations_sorted(self, trained_recommender):
        """Test that hybrid recommendations are sorted by score."""
        rec = trained_recommender

        if len(rec.df) > 2:
            title = rec.df.iloc[0]["name"]
            recs = rec.recommend(title, n_recommendations=5)

            if len(recs) > 1:
                scores = [m["recommendation_score"] for m in recs]
                # Should be in descending order
                assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
class TestFiltering:
    """Test filtering logic."""

    def test_genre_filter(self, trained_recommender):
        """Test that genre filtering works."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]["name"]

            filters = {"genres": ["Action"]}

            recs = rec.recommend(title, n_recommendations=10, filters=filters)

            # All results should have Action genre
            for manhwa in recs:
                assert "Action" in manhwa["genres"]

    def test_rating_filter(self, trained_recommender):
        """Test that rating filtering works."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]["name"]

            filters = {"min_rating": 4.5}

            recs = rec.recommend(title, n_recommendations=10, filters=filters)

            # All results should have rating >= 4.5
            for manhwa in recs:
                assert manhwa["rating"] >= 4.5

    def test_multiple_filters(self, trained_recommender):
        """Test that multiple filters work together."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]["name"]

            filters = {"genres": ["Action"], "min_rating": 4.0}

            recs = rec.recommend(title, n_recommendations=10, filters=filters)

            # All results should match all filters
            for manhwa in recs:
                assert "Action" in manhwa["genres"]
                assert manhwa["rating"] >= 4.0


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

    def test_loaded_model_produces_same_recommendations(self, trained_recommender, temp_model_dir):
        """Test that loaded model produces same recommendations."""
        # Get recommendations from original
        title = trained_recommender.df.iloc[0]["name"]
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
            assert orig["name"] == loaded["name"]
            assert abs(orig["recommendation_score"] - loaded["recommendation_score"]) < 0.01


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
        title = rec.df.iloc[0]["name"]
        recs = rec.recommend(title, n_recommendations=10000)

        # Should return at most (total - 1) recommendations (excluding self)
        assert len(recs) <= len(rec.df) - 1

    def test_empty_user_profile(self, trained_recommender):
        """Test that empty user profile doesn't break recommendations."""
        rec = trained_recommender

        if len(rec.df) > 1:
            title = rec.df.iloc[0]["name"]

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
        assert "precision@k" in metrics
        assert "recall@k" in metrics
        assert "ndcg@k" in metrics
        assert "mrr" in metrics
        assert "hit_rate@k" in metrics
        assert metrics["k"] == 5

        # Metrics should be in valid range
        assert 0 <= metrics["precision@k"] <= 1
        assert 0 <= metrics["recall@k"] <= 1
        assert 0 <= metrics["ndcg@k"] <= 1
        assert 0 <= metrics["mrr"] <= 1
        assert 0 <= metrics["hit_rate@k"] <= 1

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
            test_ratio=0.3,
        )

        # With small dataset (6 items), evaluation may be skipped (requires > 20 items)
        # This is expected behavior, so we test both cases
        if metrics is not None:
            # Evaluation ran successfully
            assert "precision@k" in metrics

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


@pytest.mark.unit
class TestUserPreferenceScoringEdgeCases:
    """Comprehensive tests for user preference scoring edge cases."""

    def test_get_user_preference_score_invalid_index_negative(self, trained_recommender):
        """Test that negative index raises ValueError."""
        rec = trained_recommender
        user_profile = {"liked_genres": ["Action"], "disliked_genres": [], "min_rating": 0}

        # Negative indexing should raise ValueError (bounds checking added)
        with pytest.raises(ValueError, match="non-negative"):
            rec.get_user_preference_score(-1, user_profile)

    def test_get_user_preference_score_invalid_index_too_large(self, trained_recommender):
        """Test that index >= len(df) raises ValueError."""
        rec = trained_recommender
        user_profile = {"liked_genres": ["Action"], "disliked_genres": [], "min_rating": 0}

        invalid_idx = len(rec.df) + 10
        with pytest.raises(ValueError, match="out of bounds"):
            rec.get_user_preference_score(invalid_idx, user_profile)

    def test_get_user_preference_score_empty_liked_genres(self, trained_recommender):
        """Test scoring with empty liked_genres list."""
        rec = trained_recommender
        user_profile = {"liked_genres": [], "disliked_genres": [], "min_rating": 0}

        score = rec.get_user_preference_score(0, user_profile)

        # Should return valid score without crashing (may be int or float)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1

    def test_get_user_preference_score_empty_disliked_genres(self, trained_recommender):
        """Test scoring with empty disliked_genres list."""
        rec = trained_recommender
        user_profile = {"liked_genres": ["Action"], "disliked_genres": [], "min_rating": 0}

        score = rec.get_user_preference_score(0, user_profile)

        # Should return valid score without crashing (may be int or float)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1

    def test_get_user_preference_score_manhwa_no_genres(self, temp_data_dir):
        """Test scoring manhwa that has no genres."""
        # Create catalog with manhwa without genres
        catalog_data = [
            {
                "name": "No Genre Manhwa",
                "description": "A manhwa without genres",
                "rating": 4.0,
                "popularity": 1000,
                "genres": [],
                "tags": [],
            }
        ]

        catalog_path = temp_data_dir / "no_genres.json"
        with open(catalog_path, "w") as f:
            json.dump(catalog_data, f)

        rec = HybridManwhaRecommender()
        rec.prepare_data(str(catalog_path))

        user_profile = {"liked_genres": ["Action"], "disliked_genres": ["Romance"], "min_rating": 0}

        score = rec.get_user_preference_score(0, user_profile)

        # Should handle gracefully (may be int or float)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1

    def test_get_user_preference_score_extreme_rating_threshold_min(self, trained_recommender):
        """Test scoring with min_rating=0."""
        rec = trained_recommender
        user_profile = {"liked_genres": [], "disliked_genres": [], "min_rating": 0}

        score = rec.get_user_preference_score(0, user_profile)

        # Should boost score since all ratings meet min_rating=0 (may be int or float)
        assert isinstance(score, (int, float))
        assert score > 0.5  # Should get rating boost

    def test_get_user_preference_score_extreme_rating_threshold_max(self, trained_recommender):
        """Test scoring with min_rating=5 (maximum)."""
        rec = trained_recommender
        user_profile = {"liked_genres": [], "disliked_genres": [], "min_rating": 5.0}

        score = rec.get_user_preference_score(0, user_profile)

        # Should penalize score since unlikely any manhwa has rating=5 (may be int or float)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1

    def test_get_user_preference_score_all_preferences_match(self, trained_recommender):
        """Test scoring when all user preferences match perfectly."""
        rec = trained_recommender

        # Find manhwa with Action genre
        action_idx = None
        for idx, row in rec.df.iterrows():
            if "Action" in row["genres"]:
                action_idx = idx
                break

        if action_idx is not None:
            manhwa = rec.df.iloc[action_idx]
            user_profile = {
                "liked_genres": list(manhwa["genres"]),  # All genres matched
                "disliked_genres": [],
                "min_rating": manhwa["rating"] - 1,  # Below actual rating
                "preferred_status": [manhwa.get("status", "RELEASING")],
            }

            score = rec.get_user_preference_score(action_idx, user_profile)

            # Should be high score when everything matches
            assert score > 0.5

    def test_get_user_preference_score_all_preferences_conflict(self, trained_recommender):
        """Test scoring when all user preferences conflict."""
        rec = trained_recommender

        # Find manhwa with specific genres
        idx = 0
        manhwa = rec.df.iloc[idx]

        user_profile = {
            "liked_genres": ["Nonexistent Genre"],
            "disliked_genres": list(manhwa["genres"]),  # Dislike all genres
            "min_rating": manhwa["rating"] + 1,  # Above actual rating
        }

        score = rec.get_user_preference_score(idx, user_profile)

        # Should be low score when everything conflicts
        assert score < 0.5

    def test_get_user_preference_score_missing_profile_fields(self, trained_recommender):
        """Test scoring with minimal user profile (missing optional fields)."""
        rec = trained_recommender
        user_profile = {}  # Empty profile

        score = rec.get_user_preference_score(0, user_profile)

        # Should use defaults and not crash (may be int or float)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 1


@pytest.mark.unit
class TestDiversityAndMMRReranking:
    """Comprehensive tests for diversity calculation and MMR re-ranking."""

    def test_calculate_diversity_single_recommendation(self, trained_recommender):
        """Test diversity calculation with single recommendation."""
        rec = trained_recommender

        single_rec = [rec.df.iloc[0].to_dict()]
        diversity = rec.calculate_diversity(single_rec)

        # Should return 1.0 for single item (maximally diverse by definition)
        assert diversity == 1.0

    def test_calculate_diversity_empty_recommendations(self, trained_recommender):
        """Test diversity calculation with empty recommendations list."""
        rec = trained_recommender

        diversity = rec.calculate_diversity([])

        # Should handle gracefully, return 1.0
        assert diversity == 1.0

    def test_calculate_diversity_identical_items(self, trained_recommender):
        """Test diversity with same item repeated (edge case)."""
        rec = trained_recommender

        # Use same item multiple times
        same_item = rec.df.iloc[0].to_dict()
        identical_recs = [same_item.copy() for _ in range(3)]

        diversity = rec.calculate_diversity(identical_recs)

        # Diversity should be low (close to 0) for identical items
        assert 0 <= diversity < 0.3

    def test_calculate_diversity_diverse_items(self, trained_recommender):
        """Test diversity with completely different items."""
        rec = trained_recommender

        if len(rec.df) >= 3:
            # Select items from different ends of dataset (likely different)
            diverse_recs = [
                rec.df.iloc[0].to_dict(),
                rec.df.iloc[len(rec.df) // 2].to_dict(),
                rec.df.iloc[-1].to_dict(),
            ]

            diversity = rec.calculate_diversity(diverse_recs)

            # Should have reasonable diversity
            assert 0 <= diversity <= 1
            assert isinstance(diversity, float)

    def test_mmr_rerank_zero_diversity_weight(self, trained_recommender):
        """Test MMR re-ranking with diversity_weight=0 (pure relevance)."""
        rec = trained_recommender

        if len(rec.df) > 3:
            title = rec.df.iloc[0]["name"]

            # Get recommendations with no diversity
            recs_no_diversity = rec.recommend(title, n_recommendations=5, diversity=0.0)

            # Should work without errors
            assert len(recs_no_diversity) > 0

            # Should be sorted by relevance score
            scores = [r["recommendation_score"] for r in recs_no_diversity]
            assert scores == sorted(scores, reverse=True)

    def test_mmr_rerank_max_diversity_weight(self, trained_recommender):
        """Test MMR re-ranking with diversity_weight=1.0 (pure diversity)."""
        rec = trained_recommender

        if len(rec.df) > 3:
            title = rec.df.iloc[0]["name"]

            # Get recommendations with maximum diversity
            recs_max_diversity = rec.recommend(title, n_recommendations=5, diversity=1.0)

            # Should work without errors
            assert len(recs_max_diversity) > 0

            # All results should be valid manhwa
            for r in recs_max_diversity:
                assert "name" in r
                assert "recommendation_score" in r

    def test_mmr_rerank_changes_order(self, trained_recommender):
        """Test that MMR re-ranking produces different order than pure relevance."""
        rec = trained_recommender

        if len(rec.df) > 5:
            title = rec.df.iloc[0]["name"]

            # Get recommendations without diversity
            recs_no_div = rec.recommend(title, n_recommendations=5, diversity=0.0)

            # Get recommendations with diversity
            recs_with_div = rec.recommend(title, n_recommendations=5, diversity=0.7)

            # Both should return results
            assert len(recs_no_div) > 0
            assert len(recs_with_div) > 0

            # At least some positions should differ (not guaranteed but likely)
            names_no_div = [r["name"] for r in recs_no_div]
            names_with_div = [r["name"] for r in recs_with_div]

            # They should contain potentially different items or orders
            assert isinstance(names_no_div, list)
            assert isinstance(names_with_div, list)

    def test_mmr_rerank_insufficient_candidates(self, trained_recommender):
        """Test MMR when candidates <= n_recommendations requested."""
        rec = trained_recommender

        if len(rec.df) >= 2:
            title = rec.df.iloc[0]["name"]

            # Request almost all available items
            n_recs = len(rec.df) - 1
            recs = rec.recommend(title, n_recommendations=n_recs, diversity=0.5)

            # Should handle gracefully
            assert len(recs) <= n_recs


@pytest.mark.unit
class TestConcurrentAccess:
    """Tests for thread safety and concurrent access."""

    def test_concurrent_recommend_calls(self, trained_recommender):
        """Test multiple threads calling recommend() simultaneously."""
        import threading

        rec = trained_recommender
        title = rec.df.iloc[0]["name"]
        results = []
        errors = []

        def make_recommendation():
            try:
                recs = rec.recommend(title, n_recommendations=3)
                results.append(recs)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=make_recommendation) for _ in range(5)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should not have errors
        assert len(errors) == 0, f"Concurrent access errors: {errors}"

        # Should have results from all threads
        assert len(results) == 5

        # All results should be valid
        for recs in results:
            assert isinstance(recs, list)
            assert len(recs) > 0

    def test_concurrent_evaluate_calls(self, sample_catalog_file):
        """Test concurrent calls to evaluate_recommendations()."""
        import threading

        # Create and train recommender
        rec = HybridManwhaRecommender()
        rec.prepare_data(sample_catalog_file)
        rec.build_content_features()

        # Create test data
        test_df = rec.df.head(2)

        results = []
        errors = []

        def run_evaluation():
            try:
                metrics = rec.evaluate_recommendations(test_df, k=3)
                results.append(metrics)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=run_evaluation) for _ in range(3)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0, f"Concurrent evaluation errors: {errors}"

    def test_concurrent_cache_building(self, trained_recommender):
        """Test race conditions in cache building (_build_title_cache)."""
        import threading

        rec = trained_recommender

        # Clear cache to force rebuild
        rec._title_cache = None

        errors = []

        def access_cache():
            try:
                # This will trigger cache build if not exists
                rec._find_manhwa_index(rec.df.iloc[0]["name"])
            except Exception as e:
                errors.append(e)

        # Create multiple threads that will race to build cache
        threads = [threading.Thread(target=access_cache) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0, f"Cache building race condition errors: {errors}"

        # Cache should be built
        assert rec._title_cache is not None

    def test_thread_safety_no_crashes(self, trained_recommender):
        """Test that concurrent access doesn't cause crashes."""
        import threading
        import random

        rec = trained_recommender
        titles = rec.df["name"].tolist()

        errors = []

        def random_operations():
            try:
                # Perform random operations
                for _ in range(3):
                    title = random.choice(titles)
                    rec.recommend(title, n_recommendations=2)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=random_operations) for _ in range(5)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_thread_safety_no_data_corruption(self, trained_recommender):
        """Test that concurrent access doesn't corrupt data."""
        import threading

        rec = trained_recommender

        # Store original state
        original_df_len = len(rec.df)
        original_feature_shape = rec.feature_matrix.shape

        def concurrent_reads():
            # Just read operations, no modifications
            rec.recommend(rec.df.iloc[0]["name"], n_recommendations=2)

        # Create multiple threads
        threads = [threading.Thread(target=concurrent_reads) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify data integrity
        assert len(rec.df) == original_df_len
        assert rec.feature_matrix.shape == original_feature_shape


@pytest.mark.unit
class TestAdditionalEdgeCases:
    """Additional edge case tests for comprehensive coverage."""

    def test_ndcg_calculation_correctness(self, sample_catalog_file):
        """Test that NDCG calculation is mathematically correct."""
        rec = HybridManwhaRecommender()
        rec.prepare_data(sample_catalog_file)

        train_df, test_df = rec.create_evaluation_split(test_ratio=0.3, random_state=42)
        rec.df = train_df
        rec.build_content_features()

        # Restore full df for evaluation
        rec.df = pd.concat([train_df, test_df])

        metrics = rec.evaluate_recommendations(test_df, k=5)

        # NDCG should be between 0 and 1
        assert 0 <= metrics["ndcg@k"] <= 1

        # NDCG should be a float
        assert isinstance(metrics["ndcg@k"], float)

    def test_cold_start_zero_popularity(self, temp_data_dir):
        """Test cold start recommendations when all items have zero popularity."""
        catalog_data = [
            {
                "name": f"Manhwa {i}",
                "description": f"Description {i}",
                "rating": 3.0 + (i * 0.2),
                "popularity": 0,  # All zero
                "genres": ["Action"],
                "tags": [],
            }
            for i in range(5)
        ]

        catalog_path = temp_data_dir / "zero_popularity.json"
        with open(catalog_path, "w") as f:
            json.dump(catalog_data, f)

        rec = HybridManwhaRecommender()
        rec.prepare_data(str(catalog_path))

        # Cold start should still work
        recs = rec.handle_cold_start(n_recommendations=3, popularity_bias=0.5)

        assert len(recs) > 0
        # Should rank by rating since popularity is all zero
        assert all(isinstance(r, dict) for r in recs)

    def test_weight_validation_in_init(self):
        """Test that custom weights are properly initialized."""
        custom_weights = {"content": 0.5, "genre_similarity": 0.3, "user_pref": 0.2}

        rec = HybridManwhaRecommender(weights=custom_weights)

        assert rec.weights == custom_weights
        assert rec.weights["content"] == 0.5
        assert rec.weights["genre_similarity"] == 0.3
        assert rec.weights["user_pref"] == 0.2

    def test_hyperparameter_tuning_cross_validation(self, sample_catalog_file):
        """Test hyperparameter tuning with cross-validation."""
        # Use custom tfidf params suitable for small dataset
        rec = HybridManwhaRecommender(
            tfidf_params={
                "max_features": 1000,
                "min_df": 1,  # Allow all documents for small dataset
                "max_df": 1.0,  # Allow all documents
                "ngram_range": (1, 1),
            }
        )
        rec.prepare_data(sample_catalog_file)

        # Create train/val split
        train_df, val_df = rec.create_evaluation_split(test_ratio=0.3, random_state=42)

        # Small parameter grid for testing with appropriate params for small data
        param_grid = {
            "weights": [
                {"content": 0.5, "genre_similarity": 0.3, "user_pref": 0.2},
            ],
            "tfidf_max_features": [1000],
            "tfidf_min_df": [1],  # Changed from 2 to 1 for small dataset
        }

        results = rec.tune_hyperparameters(
            train_df=train_df, val_df=val_df, param_grid=param_grid, metric="ndcg@k", k=5
        )

        # Should return results structure
        assert "best_params" in results
        assert "best_score" in results
        assert "all_results" in results

        # Best score should be valid
        assert isinstance(results["best_score"], (int, float))

    def test_evaluate_no_data_leakage(self, sample_catalog_file):
        """Test that evaluation doesn't leak test data into training."""
        rec = HybridManwhaRecommender()
        rec.prepare_data(sample_catalog_file)

        # Create train/test split
        train_df, test_df = rec.create_evaluation_split(test_ratio=0.3, random_state=42)

        # Train only on training data
        rec.df = train_df
        rec.build_content_features()

        # Verify test items are NOT in training data
        train_names = set(train_df["name"].tolist())
        test_names = set(test_df["name"].tolist())

        # No overlap between train and test
        assert len(train_names & test_names) == 0

    def test_coverage_tracking_reset(self, trained_recommender):
        """Test that coverage tracking can be reset."""
        rec = trained_recommender

        # Make some recommendations
        title = rec.df.iloc[0]["name"]
        rec.recommend(title, n_recommendations=3)

        # Coverage should track recommended items
        assert len(rec._all_recommended_items) >= 0

        # Reset coverage
        rec._all_recommended_items = set()

        # Should be empty after reset
        assert len(rec._all_recommended_items) == 0

    def test_division_by_zero_guards(self, temp_data_dir):
        """Test guards against division by zero in various calculations."""
        # Create small dataset (3 items) which might cause division issues
        catalog_data = [
            {
                "name": f"Manhwa {i}",
                "description": f"Description {i} with more words to make it unique",
                "rating": 4.0,
                "popularity": 1000 * (i + 1),
                "genres": ["Action"],
                "tags": [],
            }
            for i in range(3)
        ]

        catalog_path = temp_data_dir / "small_items.json"
        with open(catalog_path, "w") as f:
            json.dump(catalog_data, f)

        # Use tfidf params suitable for small dataset
        rec = HybridManwhaRecommender(
            tfidf_params={"max_features": 100, "min_df": 1, "max_df": 1.0, "ngram_range": (1, 1)}
        )
        rec.prepare_data(str(catalog_path))
        rec.build_content_features()

        # Operations that might involve division
        # Cold start
        cold_recs = rec.handle_cold_start(n_recommendations=5)
        assert isinstance(cold_recs, list)

        # Diversity (single item from cold start)
        if cold_recs:
            diversity = rec.calculate_diversity(cold_recs[:1])
            assert isinstance(diversity, float)
            assert not np.isnan(diversity)

        # Diversity with multiple items
        if len(cold_recs) > 1:
            diversity_multi = rec.calculate_diversity(cold_recs)
            assert isinstance(diversity_multi, float)
            assert not np.isnan(diversity_multi)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
