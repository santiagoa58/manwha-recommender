"""
Comprehensive tests for the deduplication system.
Tests actual functionality including fuzzy matching, merging logic, and data transformations.
"""

import pytest
from src.data_processing.deduplicator import ManwhaDeduplicator


class TestTitleNormalization:
    """Test title normalization logic."""

    def test_normalize_basic_title(self):
        """Test normalization of a basic title."""
        dedup = ManwhaDeduplicator()
        result = dedup.normalize_title("Solo Leveling")
        assert result == "solo leveling"

    def test_normalize_removes_parentheses(self):
        """Test that parentheses content is removed."""
        dedup = ManwhaDeduplicator()
        result = dedup.normalize_title("Tower of God (Season 1)")
        assert result == "tower of god"

    def test_normalize_removes_brackets(self):
        """Test that bracket content is removed."""
        dedup = ManwhaDeduplicator()
        result = dedup.normalize_title("Solo Leveling [Official]")
        assert result == "solo leveling"

    def test_normalize_removes_manhwa_suffix(self):
        """Test that 'manhwa' suffix is removed."""
        dedup = ManwhaDeduplicator()
        result = dedup.normalize_title("Tower of God Manhwa")
        assert result == "tower of god"

    def test_normalize_removes_part_suffixes(self):
        """Test that part/season suffixes are removed."""
        dedup = ManwhaDeduplicator()
        assert dedup.normalize_title("Story Part 2") == "story"
        assert dedup.normalize_title("Series Season 3") == "series"
        assert dedup.normalize_title("Title S2") == "title"

    def test_normalize_collapses_whitespace(self):
        """Test that multiple spaces are collapsed."""
        dedup = ManwhaDeduplicator()
        result = dedup.normalize_title("Solo    Leveling   Part    1")
        assert "    " not in result
        assert result == "solo leveling"

    def test_normalize_handles_empty_string(self):
        """Test that empty strings are handled."""
        dedup = ManwhaDeduplicator()
        result = dedup.normalize_title("")
        assert result == ""

    def test_normalize_handles_none(self):
        """Test that None is handled."""
        dedup = ManwhaDeduplicator()
        result = dedup.normalize_title(None)
        assert result == ""


@pytest.mark.unit
class TestDuplicateDetection:
    """Test duplicate detection logic."""

    def test_exact_match_detection(self, duplicate_manhwa_entries):
        """Test that exact title matches are detected."""
        dedup = ManwhaDeduplicator()

        # Filter to only Solo Leveling entries
        solo_entries = [e for e in duplicate_manhwa_entries if "Solo" in e["name"]]

        duplicate_groups = dedup.find_duplicates(solo_entries)

        # Should find 1 group with 3 Solo Leveling entries
        assert len(duplicate_groups) > 0
        solo_group = [g for g in duplicate_groups if len(g) == 3]
        assert len(solo_group) == 1
        assert len(solo_group[0]) == 3

    def test_no_duplicates_in_unique_set(self, sample_manhwa_list):
        """Test that unique entries don't produce duplicate groups."""
        dedup = ManwhaDeduplicator()
        duplicate_groups = dedup.find_duplicates(sample_manhwa_list)

        # Should find no duplicates (all unique titles)
        assert len(duplicate_groups) == 0

    def test_fuzzy_match_similar_titles(self):
        """Test that similar titles are matched."""
        dedup = ManwhaDeduplicator()
        entries = [
            {"id": "1", "name": "Solo Leveling", "source": "A"},
            {"id": "2", "name": "Solo  Leveling", "source": "B"},  # Extra spaces
            {
                "id": "3",
                "name": "Solo Leveling Part 1",
                "source": "C",
            },  # With suffix that gets removed
        ]

        duplicate_groups = dedup.find_duplicates(entries)

        # All three should be in same group (after normalization)
        assert len(duplicate_groups) == 1
        assert len(duplicate_groups[0]) >= 2  # At least 2 should match

    def test_different_titles_not_matched(self):
        """Test that different titles are not matched as duplicates."""
        dedup = ManwhaDeduplicator()
        entries = [
            {"id": "1", "name": "Solo Leveling", "source": "A"},
            {"id": "2", "name": "Tower of God", "source": "B"},
            {"id": "3", "name": "The Beginning After The End", "source": "C"},
        ]

        duplicate_groups = dedup.find_duplicates(entries)

        # Should find no duplicates
        assert len(duplicate_groups) == 0

    def test_partial_match_above_threshold(self):
        """Test that titles above similarity threshold are matched."""
        dedup = ManwhaDeduplicator()
        dedup.TITLE_SIMILARITY_THRESHOLD = 80  # Lower threshold for test

        entries = [
            {"id": "1", "name": "The Beginning After The End", "source": "A"},
            {"id": "2", "name": "Beginning After The End", "source": "B"},  # Similar
        ]

        duplicate_groups = dedup.find_duplicates(entries)

        # Should match as duplicates
        assert len(duplicate_groups) == 1


@pytest.mark.unit
class TestEntryMerging:
    """Test entry merging logic."""

    def test_merge_preserves_highest_priority_source(self, duplicate_manhwa_entries):
        """Test that merging prefers MangaUpdates > AniList > MAL."""
        dedup = ManwhaDeduplicator()

        # Get Solo Leveling entries
        solo_entries = [e for e in duplicate_manhwa_entries if "Solo" in e["name"]]

        merged = dedup._merge_group(solo_entries)

        # Should use MangaUpdates description (most detailed)
        assert merged["description"] == "Most detailed description here"

    def test_merge_combines_alternative_names(self, duplicate_manhwa_entries):
        """Test that alternative names from all sources are combined."""
        dedup = ManwhaDeduplicator()

        solo_entries = [e for e in duplicate_manhwa_entries if "Solo" in e["name"]]
        merged = dedup._merge_group(solo_entries)

        # Should include alt names from multiple sources
        alt_names = merged["altName"]
        assert "Na Honjaman Level-Up" in alt_names or "Only I Level Up" in alt_names

    def test_merge_aggregates_ratings_weighted(self, duplicate_manhwa_entries):
        """Test that ratings are aggregated with weighted average."""
        dedup = ManwhaDeduplicator()

        solo_entries = [e for e in duplicate_manhwa_entries if "Solo" in e["name"]]
        merged = dedup._merge_group(solo_entries)

        # Should be weighted average favoring higher popularity
        # MU has highest popularity (200k) and rating (4.9)
        assert merged["rating"] >= 4.8  # Should be close to higher ratings
        assert merged["rating"] <= 4.9

    def test_merge_unions_genres(self, duplicate_manhwa_entries):
        """Test that genres from all sources are combined."""
        dedup = ManwhaDeduplicator()

        solo_entries = [e for e in duplicate_manhwa_entries if "Solo" in e["name"]]
        merged = dedup._merge_group(solo_entries)

        # Should have genres from all sources
        genres = set(merged["genres"])
        assert "Action" in genres
        assert "Fantasy" in genres or "Adventure" in genres or "Supernatural" in genres
        assert len(genres) >= 3  # Should have combined from multiple sources

    def test_merge_unions_tags(self, duplicate_manhwa_entries):
        """Test that tags from all sources are combined."""
        dedup = ManwhaDeduplicator()

        solo_entries = [e for e in duplicate_manhwa_entries if "Solo" in e["name"]]
        merged = dedup._merge_group(solo_entries)

        # Should have tags from all sources
        tags = set(merged["tags"])
        assert "Dungeon" in tags or "Game" in tags

    def test_merge_tracks_sources(self, duplicate_manhwa_entries):
        """Test that merge tracks all contributing sources."""
        dedup = ManwhaDeduplicator()

        solo_entries = [e for e in duplicate_manhwa_entries if "Solo" in e["name"]]
        merged = dedup._merge_group(solo_entries)

        # Should list all sources
        assert "sources" in merged
        assert len(merged["sources"]) == 3
        assert "AniList" in merged["sources"]
        assert "MyAnimeList" in merged["sources"]
        assert "MangaUpdates" in merged["sources"]

    def test_merge_uses_max_popularity(self, duplicate_manhwa_entries):
        """Test that merge uses maximum popularity value."""
        dedup = ManwhaDeduplicator()

        solo_entries = [e for e in duplicate_manhwa_entries if "Solo" in e["name"]]
        merged = dedup._merge_group(solo_entries)

        # Should use max popularity (200000 from MangaUpdates)
        assert merged["popularity"] == 200000


@pytest.mark.unit
class TestRatingAggregation:
    """Test rating aggregation with different weights."""

    def test_weighted_rating_calculation(self):
        """Test that weighted rating calculation works correctly."""
        dedup = ManwhaDeduplicator()

        entries = [
            {"id": "1", "name": "Test", "rating": 5.0, "popularity": 100, "source": "A"},
            {"id": "2", "name": "Test", "rating": 3.0, "popularity": 10, "source": "B"},
        ]

        merged = dedup._merge_group(entries)

        # Weighted average should favor the entry with more popularity
        # (5.0 * 100 + 3.0 * 10) / (100 + 10) = 530 / 110 ≈ 4.82
        assert merged["rating"] >= 4.7
        assert merged["rating"] <= 4.9

    def test_equal_weight_rating(self):
        """Test rating with equal weights."""
        dedup = ManwhaDeduplicator()

        entries = [
            {"id": "1", "name": "Test", "rating": 4.0, "popularity": 100, "source": "A"},
            {"id": "2", "name": "Test", "rating": 5.0, "popularity": 100, "source": "B"},
        ]

        merged = dedup._merge_group(entries)

        # Should be simple average: (4.0 + 5.0) / 2 = 4.5
        assert merged["rating"] == pytest.approx(4.5, abs=0.1)

    def test_missing_ratings_handled(self):
        """Test that entries without ratings don't break aggregation."""
        dedup = ManwhaDeduplicator()

        entries = [
            {"id": "1", "name": "Test", "rating": 4.5, "popularity": 100, "source": "A"},
            {"id": "2", "name": "Test", "source": "B"},  # No rating
        ]

        merged = dedup._merge_group(entries)

        # Should still have a rating from the first entry
        assert merged["rating"] == 4.5


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases in deduplication."""

    def test_empty_input_list(self):
        """Test that empty input is handled."""
        dedup = ManwhaDeduplicator()
        duplicate_groups = dedup.find_duplicates([])
        assert duplicate_groups == []

    def test_single_entry(self):
        """Test that single entry produces no duplicates."""
        dedup = ManwhaDeduplicator()
        entries = [{"id": "1", "name": "Solo Leveling", "source": "A"}]
        duplicate_groups = dedup.find_duplicates(entries)
        assert len(duplicate_groups) == 0

    def test_all_entries_duplicates(self):
        """Test when all entries are duplicates."""
        dedup = ManwhaDeduplicator()
        entries = [
            {"id": "1", "name": "Solo Leveling", "source": "A"},
            {"id": "2", "name": "Solo Leveling", "source": "B"},
            {"id": "3", "name": "Solo Leveling", "source": "C"},
        ]

        duplicate_groups = dedup.find_duplicates(entries)

        # Should have 1 group with all 3 entries
        assert len(duplicate_groups) == 1
        assert len(duplicate_groups[0]) == 3

    def test_missing_name_field(self):
        """Test that entries without name field are handled."""
        dedup = ManwhaDeduplicator()
        entries = [
            {"id": "1", "name": "Solo Leveling", "source": "A"},
            {"id": "2", "source": "B"},  # No name
        ]

        # Should not crash
        duplicate_groups = dedup.find_duplicates(entries)
        assert isinstance(duplicate_groups, list)

    def test_unicode_titles(self):
        """Test that unicode/non-English titles are handled."""
        dedup = ManwhaDeduplicator()
        entries = [
            {"id": "1", "name": "나 혼자만 레벨업", "source": "A"},
            {"id": "2", "name": "나 혼자만 레벨업", "source": "B"},
        ]

        duplicate_groups = dedup.find_duplicates(entries)

        # Should match unicode titles
        assert len(duplicate_groups) == 1


@pytest.mark.unit
class TestProcessAllSources:
    """Test the complete deduplication pipeline."""

    def test_process_multiple_sources(self, sample_manhwa_list):
        """Test processing entries from multiple sources."""
        dedup = ManwhaDeduplicator()

        # Create duplicates across sources
        anilist_data = [sample_manhwa_list[0]]
        jikan_data = [sample_manhwa_list[1]]
        mu_data = [sample_manhwa_list[2]]
        ap_data = []

        result = dedup.process_all_sources(
            anilist_data=anilist_data,
            jikan_data=jikan_data,
            mangaupdates_data=mu_data,
            animeplanet_data=ap_data,
        )

        # Should return combined and deduplicated list
        assert len(result) == 3  # All unique
        assert all("source" in entry for entry in result)

    def test_process_with_duplicates_across_sources(self):
        """Test deduplication across different sources."""
        dedup = ManwhaDeduplicator()

        # Same manhwa from different sources
        anilist_data = [
            {
                "id": "anilist_1",
                "name": "Solo Leveling",
                "rating": 4.7,
                "genres": ["Action"],
                "tags": [],
                "source": "AniList",
            }
        ]

        jikan_data = [
            {
                "id": "mal_1",
                "name": "Solo Leveling",
                "rating": 4.8,
                "genres": ["Fantasy"],
                "tags": [],
                "source": "MyAnimeList",
            }
        ]

        result = dedup.process_all_sources(
            anilist_data=anilist_data,
            jikan_data=jikan_data,
            mangaupdates_data=[],
            animeplanet_data=[],
        )

        # Should merge into 1 entry
        assert len(result) == 1

        # Should have combined genres
        assert "Action" in result[0]["genres"]
        assert "Fantasy" in result[0]["genres"]

        # Should list both sources
        assert len(result[0]["sources"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
