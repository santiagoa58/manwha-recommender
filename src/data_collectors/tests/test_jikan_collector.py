"""
Tests for Jikan (MyAnimeList) API collector.
Focus on testing transformation logic with mocked API responses.
"""

import pytest
from src.data_collectors.jikan_collector import JikanCollector


@pytest.fixture
def jikan_response():
    """Sample Jikan API response."""
    return {
        "data": [
            {
                "mal_id": 123,
                "title": "Solo Leveling",
                "titles": [
                    {"type": "Default", "title": "Solo Leveling"},
                    {"type": "English", "title": "Solo Leveling"},
                    {"type": "Japanese", "title": "俺だけレベルアップな件"},
                    {"type": "Synonym", "title": "Only I Level Up"},
                ],
                "type": "Manhwa",
                "score": 9.4,
                "scored_by": 100000,
                "members": 250000,
                "favorites": 50000,
                "synopsis": "E-class hunter Jinwoo Sung...",
                "status": "Finished",
                "chapters": 200,
                "volumes": 45,
                "published": {
                    "from": "2018-03-04T00:00:00+00:00",
                    "to": "2023-01-01T00:00:00+00:00",
                },
                "genres": [{"mal_id": 1, "name": "Action"}, {"mal_id": 10, "name": "Fantasy"}],
                "themes": [{"mal_id": 31, "name": "Super Power"}],
                "demographics": [{"mal_id": 27, "name": "Shounen"}],
                "authors": [{"mal_id": 1, "name": "Chugong"}],
                "serializations": [{"mal_id": 1, "name": "Kakao Page"}],
                "images": {"jpg": {"large_image_url": "https://example.com/image.jpg"}},
                "rank": 42,
                "url": "https://myanimelist.net/manga/123",
            }
        ],
        "pagination": {"has_next_page": False, "last_visible_page": 1},
    }


@pytest.mark.unit
class TestJikanTransformation:
    """Test data transformation logic."""

    def test_transform_basic_entry(self, jikan_response):
        """Test that basic entry is transformed correctly."""
        collector = JikanCollector()

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        assert len(transformed) == 1
        entry = transformed[0]

        assert entry["id"] == "mal_123"
        assert entry["mal_id"] == 123
        assert entry["name"] == "Solo Leveling"
        assert entry["source"] == "MyAnimeList"

    def test_transform_rating_conversion(self, jikan_response):
        """Test that MAL scores are properly converted to 0-5 scale."""
        collector = JikanCollector()

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        entry = transformed[0]

        # Score is 9.4, should be 9.4/2 = 4.7
        assert entry["rating"] == 4.7

    def test_transform_alt_titles_extraction(self, jikan_response):
        """Test that alternative titles are extracted."""
        collector = JikanCollector()

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        entry = transformed[0]

        # Should include English, Japanese, and Synonym titles (but not Default)
        alt_names = entry["altName"]
        assert "俺だけレベルアップな件" in alt_names or "Only I Level Up" in alt_names

    def test_transform_alt_titles_limit(self, jikan_response):
        """Test that alt titles are limited to 3."""
        collector = JikanCollector()

        # Add many titles
        jikan_response["data"][0]["titles"].extend(
            [{"type": "Synonym", "title": f"Title {i}"} for i in range(10)]
        )

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        entry = transformed[0]

        # Should be limited to 3 alt names
        alt_count = len(entry["altName"].split(", ")) if entry["altName"] else 0
        assert alt_count <= 3

    def test_transform_genres_themes_demographics(self, jikan_response):
        """Test that genres, themes, and demographics are combined."""
        collector = JikanCollector()

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        entry = transformed[0]

        # Should combine all into tags
        all_tags = entry["tags"]
        assert "Action" in all_tags
        assert "Fantasy" in all_tags
        assert "Super Power" in all_tags
        assert "Shounen" in all_tags

        # Should also have separate genre list
        assert "Action" in entry["genres"]
        assert "Fantasy" in entry["genres"]

    def test_transform_status_mapping(self, jikan_response):
        """Test that status is properly mapped."""
        collector = JikanCollector()

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        entry = transformed[0]

        # "Finished" should map to "FINISHED"
        assert entry["status"] == "FINISHED"

    def test_transform_status_publishing(self):
        """Test status mapping for ongoing series."""
        collector = JikanCollector()

        entries = [
            {
                "mal_id": 1,
                "title": "Test",
                "titles": [],
                "type": "Manhwa",
                "status": "Publishing",
                "genres": [],
                "themes": [],
                "demographics": [],
            }
        ]

        transformed = collector._transform_entries(entries)
        entry = transformed[0]

        assert entry["status"] == "RELEASING"

    def test_transform_status_hiatus(self):
        """Test status mapping for hiatus."""
        collector = JikanCollector()

        entries = [
            {
                "mal_id": 1,
                "title": "Test",
                "titles": [],
                "type": "Manhwa",
                "status": "On Hiatus",
                "genres": [],
                "themes": [],
                "demographics": [],
            }
        ]

        transformed = collector._transform_entries(entries)
        entry = transformed[0]

        assert entry["status"] == "HIATUS"

    def test_transform_date_formatting(self, jikan_response):
        """Test that dates are properly formatted."""
        collector = JikanCollector()

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        entry = transformed[0]

        assert entry["start_date"] == "2018-03-04"
        assert entry["end_date"] == "2023-01-01"
        assert entry["years"] == "2018 - 2023"

    def test_transform_ongoing_date_formatting(self):
        """Test date formatting for ongoing series."""
        collector = JikanCollector()

        entries = [
            {
                "mal_id": 1,
                "title": "Ongoing Series",
                "titles": [],
                "type": "Manhwa",
                "published": {"from": "2020-01-01T00:00:00+00:00", "to": None},
                "genres": [],
                "themes": [],
                "demographics": [],
            }
        ]

        transformed = collector._transform_entries(entries)
        entry = transformed[0]

        assert entry["start_date"] == "2020-01-01"
        assert entry["end_date"] is None
        assert entry["years"] == "2020 - Ongoing"

    def test_transform_authors_extraction(self, jikan_response):
        """Test that authors are extracted."""
        collector = JikanCollector()

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        entry = transformed[0]

        assert len(entry["authors"]) > 0
        assert "Chugong" in entry["authors"][0]

    def test_transform_serializations_extraction(self, jikan_response):
        """Test that serializations are extracted."""
        collector = JikanCollector()

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        entry = transformed[0]

        assert len(entry["serializations"]) > 0
        assert "Kakao Page" in entry["serializations"][0]

    def test_transform_popularity_metrics(self, jikan_response):
        """Test that popularity metrics are extracted."""
        collector = JikanCollector()

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        entry = transformed[0]

        assert entry["popularity"] == 250000  # members count
        assert entry["favourites"] == 50000
        assert entry["scored_by"] == 100000

    def test_transform_missing_score(self):
        """Test handling when score is missing."""
        collector = JikanCollector()

        entries = [
            {
                "mal_id": 1,
                "title": "No Score",
                "titles": [],
                "type": "Manhwa",
                "score": None,
                "genres": [],
                "themes": [],
                "demographics": [],
            }
        ]

        transformed = collector._transform_entries(entries)
        entry = transformed[0]

        assert entry["rating"] is None

    def test_transform_missing_dates(self):
        """Test handling of missing dates."""
        collector = JikanCollector()

        entries = [
            {
                "mal_id": 1,
                "title": "No Dates",
                "titles": [],
                "type": "Manhwa",
                "published": {},
                "genres": [],
                "themes": [],
                "demographics": [],
            }
        ]

        transformed = collector._transform_entries(entries)
        entry = transformed[0]

        assert entry["start_date"] is None
        assert entry["end_date"] is None
        assert entry["years"] == "Unknown"

    def test_transform_missing_optional_fields(self):
        """Test that missing optional fields don't break transformation."""
        collector = JikanCollector()

        minimal_entry = [
            {
                "mal_id": 1,
                "title": "Minimal Entry",
                "titles": [],
                "type": "Manhwa",
                "genres": [],
                "themes": [],
                "demographics": [],
            }
        ]

        transformed = collector._transform_entries(minimal_entry)

        assert len(transformed) == 1
        entry = transformed[0]

        assert entry["id"] == "mal_1"
        assert entry["name"] == "Minimal Entry"
        assert entry["rating"] is None
        assert entry["popularity"] == 0

    def test_transform_chapters_volumes(self, jikan_response):
        """Test that chapters and volumes are extracted."""
        collector = JikanCollector()

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        entry = transformed[0]

        assert entry["chapters"] == 200
        assert entry["volumes"] == 45

    def test_transform_missing_chapters(self):
        """Test handling of missing chapters/volumes."""
        collector = JikanCollector()

        entries = [
            {
                "mal_id": 1,
                "title": "No Chapters",
                "titles": [],
                "type": "Manhwa",
                # chapters and volumes not present (missing key)
                "genres": [],
                "themes": [],
                "demographics": [],
            }
        ]

        transformed = collector._transform_entries(entries)
        entry = transformed[0]

        # When key is missing, defaults to "Unknown"
        assert entry["chapters"] == "Unknown"
        assert entry["volumes"] == "Unknown"

    def test_transform_image_url_extraction(self, jikan_response):
        """Test that image URL is extracted correctly."""
        collector = JikanCollector()

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        entry = transformed[0]

        assert entry["imageURL"] == "https://example.com/image.jpg"

    def test_transform_missing_image(self):
        """Test handling of missing image."""
        collector = JikanCollector()

        entries = [
            {
                "mal_id": 1,
                "title": "No Image",
                "titles": [],
                "type": "Manhwa",
                "images": {},
                "genres": [],
                "themes": [],
                "demographics": [],
            }
        ]

        transformed = collector._transform_entries(entries)
        entry = transformed[0]

        assert entry["imageURL"] is None

    def test_transform_country_detection(self, jikan_response):
        """Test that country is set based on type."""
        collector = JikanCollector()

        entries = jikan_response["data"]
        transformed = collector._transform_entries(entries)

        entry = transformed[0]

        # Type is "Manhwa", country should be "KR"
        assert entry["country"] == "KR"

    def test_transform_non_manhwa_country(self):
        """Test country for non-manhwa entries."""
        collector = JikanCollector()

        entries = [
            {
                "mal_id": 1,
                "title": "Manga",
                "titles": [],
                "type": "Manga",  # Not Manhwa
                "genres": [],
                "themes": [],
                "demographics": [],
            }
        ]

        transformed = collector._transform_entries(entries)
        entry = transformed[0]

        assert entry["country"] == "Unknown"

    def test_transform_error_handling(self):
        """Test that transformation errors are logged but don't break the pipeline."""
        collector = JikanCollector()

        # Entry with invalid data that might cause errors
        entries = [
            {
                "mal_id": 1,
                "title": "Valid Entry",
                "titles": [],
                "type": "Manhwa",
                "genres": [],
                "themes": [],
                "demographics": [],
            },
            {
                # Malformed entry without mal_id
                "title": "Broken Entry",
                "titles": [],
                "type": "Manhwa",
            },
        ]

        # Should handle error and continue
        transformed = collector._transform_entries(entries)

        # Should have at least the valid entry
        assert len(transformed) >= 1
        assert transformed[0]["name"] == "Valid Entry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
