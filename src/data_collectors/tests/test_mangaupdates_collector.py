"""
Tests for MangaUpdates API collector.
Focus on testing transformation logic with mocked API responses.
"""

import pytest
from src.data_collectors.mangaupdates_collector import MangaUpdatesCollector


@pytest.fixture
def mangaupdates_entry():
    """Sample MangaUpdates API entry."""
    return {
        "series_id": 456,
        "title": "Solo Leveling",
        "associated": [
            {"title": "Na Honjaman Level-Up"},
            {"title": "Only I Level Up"}
        ],
        "type": "Manhwa",
        "description": "E-class hunter story",
        "bayesian_rating": 9.8,
        "rating_votes": 50000,
        "year": "2018",
        "status": "Complete",
        "latest_chapter": 200,
        "genres": [
            {"genre": "Action"},
            {"genre": "Fantasy"}
        ],
        "categories": [
            {"category": "Dungeons"},
            {"category": "Overpowered MC"}
        ],
        "image": {
            "url": {
                "original": "https://example.com/image.jpg"
            }
        },
        "authors": [
            {"name": "Chugong", "type": "Story"},
            {"name": "DUBU", "type": "Art"}
        ],
        "publishers": [
            {"publisher_name": "Kakao", "type": "Original"},
            {"publisher_name": "Yen Press", "type": "English"}
        ],
        "recommendations": [
            {"series_name": "Tower of God"},
            {"series_name": "The Beginning After The End"}
        ],
        "rank": {
            "position": 12
        },
        "url": "https://www.mangaupdates.com/series/456",
        "forum_id": 789
    }


@pytest.mark.unit
class TestMangaUpdatesTransformation:
    """Test data transformation logic."""

    def test_transform_basic_entry(self, mangaupdates_entry):
        """Test that basic entry is transformed correctly."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        assert transformed["id"] == "mu_456"
        assert transformed["mangaupdates_id"] == 456
        assert transformed["name"] == "Solo Leveling"
        assert transformed["source"] == "MangaUpdates"

    def test_transform_rating_conversion(self, mangaupdates_entry):
        """Test that MU bayesian ratings are converted to 0-5 scale."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        # bayesian_rating is 9.8, should be 9.8/2 = 4.9
        assert transformed["rating"] == 4.9
        assert transformed["bayesian_rating"] == 9.8

    def test_transform_alt_titles_extraction(self, mangaupdates_entry):
        """Test that alternative titles are extracted."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        alt_names = transformed["altName"]
        assert "Na Honjaman Level-Up" in alt_names
        assert "Only I Level Up" in alt_names

    def test_transform_alt_titles_limit(self, mangaupdates_entry):
        """Test that alt titles are limited to 3."""
        collector = MangaUpdatesCollector()

        # Add many associated names
        mangaupdates_entry["associated"].extend([
            {"title": f"Alt Title {i}"} for i in range(10)
        ])

        transformed = collector._transform_entry(mangaupdates_entry)

        # Should be limited to 3
        alt_count = len(transformed["altName"].split(", "))
        assert alt_count <= 3

    def test_transform_genres_and_categories(self, mangaupdates_entry):
        """Test that genres and categories are combined into tags."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        # Should have both genres and categories
        assert "Action" in transformed["genres"]
        assert "Fantasy" in transformed["genres"]
        assert "Dungeons" in transformed["categories"]
        assert "Overpowered MC" in transformed["categories"]

        # All should be in tags
        all_tags = transformed["tags"]
        assert "Action" in all_tags
        assert "Fantasy" in all_tags
        assert "Dungeons" in all_tags

    def test_transform_status_mapping(self, mangaupdates_entry):
        """Test that status is properly mapped."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        # "Complete" should map to "FINISHED"
        assert transformed["status"] == "FINISHED"
        assert transformed["original_status"] == "Complete"

    def test_transform_status_ongoing(self):
        """Test status mapping for ongoing series."""
        collector = MangaUpdatesCollector()

        entry = {
            "series_id": 1,
            "title": "Ongoing",
            "status": "Ongoing",
            "genres": [],
            "categories": []
        }

        transformed = collector._transform_entry(entry)

        assert transformed["status"] == "RELEASING"

    def test_transform_status_hiatus(self):
        """Test status mapping for hiatus."""
        collector = MangaUpdatesCollector()

        entry = {
            "series_id": 1,
            "title": "Hiatus",
            "status": "Hiatus",
            "genres": [],
            "categories": []
        }

        transformed = collector._transform_entry(entry)

        assert transformed["status"] == "HIATUS"

    def test_transform_status_cancelled(self):
        """Test status mapping for cancelled series."""
        collector = MangaUpdatesCollector()

        entry = {
            "series_id": 1,
            "title": "Cancelled",
            "status": "Cancelled",
            "genres": [],
            "categories": []
        }

        transformed = collector._transform_entry(entry)

        assert transformed["status"] == "CANCELLED"

    def test_transform_year_formatting_complete(self, mangaupdates_entry):
        """Test year formatting for completed series."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        assert transformed["year"] == "2018"
        assert transformed["years"] == "2018 - Complete"

    def test_transform_year_formatting_ongoing(self):
        """Test year formatting for ongoing series."""
        collector = MangaUpdatesCollector()

        entry = {
            "series_id": 1,
            "title": "Ongoing",
            "year": "2020",
            "status": "Ongoing",
            "genres": [],
            "categories": []
        }

        transformed = collector._transform_entry(entry)

        assert transformed["years"] == "2020 - Ongoing"

    def test_transform_missing_year(self):
        """Test handling of missing year."""
        collector = MangaUpdatesCollector()

        entry = {
            "series_id": 1,
            "title": "No Year",
            "status": "Complete",
            "genres": [],
            "categories": []
        }

        transformed = collector._transform_entry(entry)

        assert transformed["years"] == "Unknown"

    def test_transform_authors_extraction(self, mangaupdates_entry):
        """Test that authors are extracted with roles."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        authors = transformed["authors"]
        assert len(authors) == 2
        assert "Chugong (Story)" in authors
        assert "DUBU (Art)" in authors

    def test_transform_publishers_extraction(self, mangaupdates_entry):
        """Test that publishers are extracted."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        publishers = transformed["publishers"]
        assert len(publishers) == 2
        assert "Kakao (Original)" in publishers
        assert "Yen Press (English)" in publishers

    def test_transform_recommendations_extraction(self, mangaupdates_entry):
        """Test that recommendations are extracted."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        recommendations = transformed["recommendations"]
        assert len(recommendations) == 2
        assert "Tower of God" in recommendations
        assert "The Beginning After The End" in recommendations

    def test_transform_recommendations_limit(self, mangaupdates_entry):
        """Test that recommendations are limited to 10."""
        collector = MangaUpdatesCollector()

        # Add many recommendations
        mangaupdates_entry["recommendations"].extend([
            {"series_name": f"Series {i}"} for i in range(20)
        ])

        transformed = collector._transform_entry(mangaupdates_entry)

        # Should be limited to 10
        assert len(transformed["recommendations"]) <= 10

    def test_transform_image_url_extraction(self, mangaupdates_entry):
        """Test that image URL is extracted correctly."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        assert transformed["imageURL"] == "https://example.com/image.jpg"

    def test_transform_missing_image(self):
        """Test handling of missing image."""
        collector = MangaUpdatesCollector()

        entry = {
            "series_id": 1,
            "title": "No Image",
            "genres": [],
            "categories": []
        }

        transformed = collector._transform_entry(entry)

        assert transformed["imageURL"] is None

    def test_transform_rating_votes(self, mangaupdates_entry):
        """Test that rating votes are extracted."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        assert transformed["rating_votes"] == 50000

    def test_transform_latest_chapter(self, mangaupdates_entry):
        """Test that latest chapter is extracted."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        assert transformed["latest_chapter"] == 200

    def test_transform_rank_extraction(self, mangaupdates_entry):
        """Test that rank is extracted."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        assert transformed["rank"] == 12

    def test_transform_missing_rank(self):
        """Test handling of missing rank."""
        collector = MangaUpdatesCollector()

        entry = {
            "series_id": 1,
            "title": "No Rank",
            "genres": [],
            "categories": []
        }

        transformed = collector._transform_entry(entry)

        assert transformed["rank"] is None

    def test_transform_anime_adaptation_check(self, mangaupdates_entry):
        """Test anime adaptation flag."""
        collector = MangaUpdatesCollector()

        # Entry without anime field
        transformed = collector._transform_entry(mangaupdates_entry)
        assert transformed["anime"] == False

        # Entry with anime
        mangaupdates_entry["anime"] = {"start": "2023-01-01"}
        transformed = collector._transform_entry(mangaupdates_entry)
        assert transformed["anime"] == True

    def test_transform_missing_optional_fields(self):
        """Test that missing optional fields don't break transformation."""
        collector = MangaUpdatesCollector()

        minimal_entry = {
            "series_id": 1,
            "title": "Minimal Entry",
            "genres": [],
            "categories": []
        }

        transformed = collector._transform_entry(minimal_entry)

        assert transformed is not None
        assert transformed["id"] == "mu_1"
        assert transformed["name"] == "Minimal Entry"
        assert transformed["rating"] is None

    def test_transform_missing_bayesian_rating(self):
        """Test handling when bayesian rating is missing."""
        collector = MangaUpdatesCollector()

        entry = {
            "series_id": 1,
            "title": "No Rating",
            "genres": [],
            "categories": []
        }

        transformed = collector._transform_entry(entry)

        assert transformed["rating"] is None
        assert transformed["bayesian_rating"] is None

    def test_transform_country_detection(self, mangaupdates_entry):
        """Test that country is set based on type."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        # Type is "Manhwa", country should be "KR"
        assert transformed["country"] == "KR"

    def test_transform_non_manhwa_country(self):
        """Test country for non-manhwa entries."""
        collector = MangaUpdatesCollector()

        entry = {
            "series_id": 1,
            "title": "Manga",
            "type": "Manga",
            "genres": [],
            "categories": []
        }

        transformed = collector._transform_entry(entry)

        assert transformed["country"] == "Unknown"

    def test_transform_error_handling(self):
        """Test that transformation errors return None."""
        collector = MangaUpdatesCollector()

        # Entry that will cause error (missing series_id)
        broken_entry = {
            "title": "Broken"
        }

        transformed = collector._transform_entry(broken_entry)

        # Should return None on error
        assert transformed is None

    def test_transform_forum_id(self, mangaupdates_entry):
        """Test that forum ID is extracted."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        assert transformed["forum_id"] == 789

    def test_transform_serialization_url(self, mangaupdates_entry):
        """Test that URL is stored as serialization."""
        collector = MangaUpdatesCollector()

        transformed = collector._transform_entry(mangaupdates_entry)

        assert transformed["serialization"] == "https://www.mangaupdates.com/series/456"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
