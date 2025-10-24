"""
Tests for AniList GraphQL API collector.
Focus on testing transformation logic with mocked API responses.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.data_collectors.anilist_collector import AniListCollector


@pytest.fixture
def anilist_response():
    """Sample AniList API response."""
    return {
        "data": {
            "Page": {
                "pageInfo": {
                    "total": 100,
                    "currentPage": 1,
                    "lastPage": 2,
                    "hasNextPage": True,
                    "perPage": 50
                },
                "media": [
                    {
                        "id": 123456,
                        "idMal": 789,
                        "title": {
                            "romaji": "Solo Leveling",
                            "english": "Solo Leveling",
                            "native": "나 혼자만 레벨업"
                        },
                        "description": "E-class hunter story<br>Action packed",
                        "format": "MANGA",
                        "status": "FINISHED",
                        "startDate": {"year": 2018, "month": 3, "day": 4},
                        "endDate": {"year": 2023, "month": 1, "day": 1},
                        "chapters": 200,
                        "volumes": 45,
                        "countryOfOrigin": "KR",
                        "isLicensed": True,
                        "coverImage": {
                            "large": "https://example.com/image.jpg"
                        },
                        "genres": ["Action", "Fantasy"],
                        "tags": [
                            {"name": "Dungeon", "rank": 80},
                            {"name": "OP MC", "rank": 75},
                            {"name": "Weak Tag", "rank": 50}  # Should be filtered
                        ],
                        "meanScore": 94,
                        "averageScore": 94,
                        "popularity": 250000,
                        "favourites": 50000,
                        "staff": {
                            "edges": [
                                {
                                    "node": {"name": {"full": "Chugong"}},
                                    "role": "Story & Art"
                                }
                            ]
                        }
                    }
                ]
            }
        }
    }


@pytest.mark.unit
class TestAniListTransformation:
    """Test data transformation logic."""

    def test_transform_basic_entry(self, anilist_response):
        """Test that basic entry is transformed correctly."""
        collector = AniListCollector()

        media_list = anilist_response["data"]["Page"]["media"]
        transformed = collector._transform_entries(media_list)

        assert len(transformed) == 1
        entry = transformed[0]

        assert entry["id"] == "anilist_123456"
        assert entry["mal_id"] == 789
        assert entry["name"] == "Solo Leveling"
        assert entry["source"] == "AniList"

    def test_transform_rating_conversion(self, anilist_response):
        """Test that AniList scores are properly converted to 0-5 scale."""
        collector = AniListCollector()

        media_list = anilist_response["data"]["Page"]["media"]
        transformed = collector._transform_entries(media_list)

        entry = transformed[0]

        # meanScore is 94, should be 94/20 = 4.7
        assert entry["rating"] == 4.7

    def test_transform_alt_titles(self, anilist_response):
        """Test that alternative titles are extracted."""
        collector = AniListCollector()

        media_list = anilist_response["data"]["Page"]["media"]
        transformed = collector._transform_entries(media_list)

        entry = transformed[0]

        # Should include English and native titles
        assert "나 혼자만 레벨업" in entry["altName"]

    def test_transform_description_cleaning(self, anilist_response):
        """Test that HTML tags are removed from descriptions."""
        collector = AniListCollector()

        media_list = anilist_response["data"]["Page"]["media"]
        transformed = collector._transform_entries(media_list)

        entry = transformed[0]

        # Should remove <br> tags
        assert "<br>" not in entry["description"]
        assert "\n" in entry["description"]  # Replaced with newline

    def test_transform_genres_and_tags(self, anilist_response):
        """Test that genres and high-rank tags are combined."""
        collector = AniListCollector()

        media_list = anilist_response["data"]["Page"]["media"]
        transformed = collector._transform_entries(media_list)

        entry = transformed[0]

        # Should include both genres and tags
        assert "Action" in entry["genres"] or "Action" in entry["tags"]
        assert "Fantasy" in entry["genres"] or "Fantasy" in entry["tags"]
        assert "Dungeon" in entry["tags"]
        assert "OP MC" in entry["tags"]

        # Low-rank tag should be filtered (rank < 60)
        assert "Weak Tag" not in entry["tags"]

    def test_transform_date_formatting(self, anilist_response):
        """Test that dates are properly formatted."""
        collector = AniListCollector()

        media_list = anilist_response["data"]["Page"]["media"]
        transformed = collector._transform_entries(media_list)

        entry = transformed[0]

        assert entry["start_date"] == "2018-03-04"
        assert entry["end_date"] == "2023-01-01"
        assert entry["years"] == "2018-03-04 - 2023-01-01"

    def test_transform_staff_extraction(self, anilist_response):
        """Test that staff/authors are extracted."""
        collector = AniListCollector()

        media_list = anilist_response["data"]["Page"]["media"]
        transformed = collector._transform_entries(media_list)

        entry = transformed[0]

        assert len(entry["authors"]) > 0
        assert "Chugong" in entry["authors"][0]
        assert "Story & Art" in entry["authors"][0]

    def test_transform_missing_alt_score(self):
        """Test handling when only averageScore is available."""
        collector = AniListCollector()

        media_list = [{
            "id": 1,
            "title": {"romaji": "Test"},
            "averageScore": 80,  # Only average, no mean
            "genres": [],
            "tags": [],
            "countryOfOrigin": "KR"
        }]

        transformed = collector._transform_entries(media_list)
        entry = transformed[0]

        # Should use averageScore: 80/20 = 4.0
        assert entry["rating"] == 4.0

    def test_transform_missing_dates(self):
        """Test handling of missing dates."""
        collector = AniListCollector()

        media_list = [{
            "id": 1,
            "title": {"romaji": "Test"},
            "genres": [],
            "tags": [],
            "countryOfOrigin": "KR"
            # No startDate or endDate
        }]

        transformed = collector._transform_entries(media_list)
        entry = transformed[0]

        assert entry["start_date"] is None
        assert entry["end_date"] is None
        assert entry["years"] == "Unknown"

    def test_transform_missing_optional_fields(self):
        """Test that missing optional fields don't break transformation."""
        collector = AniListCollector()

        minimal_media = [{
            "id": 1,
            "title": {"romaji": "Minimal Entry"},
            "genres": [],
            "tags": [],
            "countryOfOrigin": "KR"
        }]

        transformed = collector._transform_entries(minimal_media)

        assert len(transformed) == 1
        entry = transformed[0]

        assert entry["id"] == "anilist_1"
        assert entry["name"] == "Minimal Entry"
        assert entry["rating"] is None
        assert entry["popularity"] == 0
        assert entry["chapters"] == "Unknown"

    def test_format_date_with_year_only(self):
        """Test date formatting with only year."""
        collector = AniListCollector()

        date_obj = {"year": 2020}
        result = collector._format_date(date_obj)

        assert result == "2020"

    def test_format_date_with_year_month(self):
        """Test date formatting with year and month."""
        collector = AniListCollector()

        date_obj = {"year": 2020, "month": 5}
        result = collector._format_date(date_obj)

        assert result == "2020-05"

    def test_format_date_with_full_date(self):
        """Test date formatting with complete date."""
        collector = AniListCollector()

        date_obj = {"year": 2020, "month": 5, "day": 15}
        result = collector._format_date(date_obj)

        assert result == "2020-05-15"

    def test_format_date_none(self):
        """Test date formatting with None."""
        collector = AniListCollector()

        result = collector._format_date(None)

        assert result is None

    def test_format_date_missing_year(self):
        """Test date formatting with missing year."""
        collector = AniListCollector()

        date_obj = {"month": 5, "day": 15}
        result = collector._format_date(date_obj)

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
