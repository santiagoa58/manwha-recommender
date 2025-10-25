"""Tests for manwha_utils.py module."""

import unittest
import pandas as pd
import numpy as np
from src.utils.manwha_utils import (
    find_manwha_by_name,
    get_target_manwha,
    get_similarity_ratio,
    get_close_matches,
    clean_alt_name,
    split_alt_names,
    map_alt_names_to_list,
)
from src.utils.constants import MANWHA_NOT_FOUND, UNKNOWN


class TestSimilarityMatching(unittest.TestCase):
    """Test similarity ratio and close matching functionality."""

    def test_get_similarity_ratio_exact_match(self):
        """Test exact match returns 1.0."""
        ratio = get_similarity_ratio("Solo Leveling", "Solo Leveling")
        self.assertEqual(ratio, 1.0)

    def test_get_similarity_ratio_case_insensitive(self):
        """Test matching is case-insensitive."""
        ratio = get_similarity_ratio("Solo Leveling", "solo leveling")
        self.assertEqual(ratio, 1.0)

    def test_get_similarity_ratio_space_insensitive(self):
        """Test matching ignores spaces."""
        ratio = get_similarity_ratio("Solo Leveling", "SoloLeveling")
        self.assertEqual(ratio, 1.0)

    def test_get_similarity_ratio_partial_match(self):
        """Test partial match returns value between 0 and 1."""
        ratio = get_similarity_ratio("Solo Leveling", "Solo")
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 1.0)

    def test_get_similarity_ratio_no_match(self):
        """Test completely different strings return low ratio."""
        ratio = get_similarity_ratio("Solo Leveling", "Tower of God")
        self.assertLess(ratio, 0.5)

    def test_get_close_matches_single(self):
        """Test finding single close match."""
        sources = ["Solo Leveling", "Tower of God", "The Beginning After The End"]
        matches = get_close_matches("Solo Leveling", sources, limit=1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], "Solo Leveling")
        self.assertEqual(matches[0][1], 1.0)

    def test_get_close_matches_with_typo(self):
        """Test finding match with typo."""
        sources = ["Solo Leveling", "Tower of God", "The Beginning After The End"]
        matches = get_close_matches("Solo Levling", sources, limit=1)
        self.assertEqual(matches[0][0], "Solo Leveling")
        self.assertGreater(matches[0][1], 0.9)

    def test_get_close_matches_multiple(self):
        """Test finding multiple close matches."""
        sources = [
            "Solo Leveling",
            "Solo Leveling 2",
            "Tower of God",
            "Solo Player",
        ]
        matches = get_close_matches("Solo", sources, limit=3)
        self.assertEqual(len(matches), 3)
        # All matches should contain 'Solo'
        for match_name, _ in matches:
            self.assertIn("Solo", match_name)

    def test_get_close_matches_sorted_by_similarity(self):
        """Test matches are sorted by similarity (highest first)."""
        sources = ["Tower of God", "Solo Leveling 2", "Solo Leveling"]
        matches = get_close_matches("Solo Leveling", sources, limit=3)
        # First match should have highest similarity
        self.assertGreater(matches[0][1], matches[1][1])
        self.assertGreater(matches[1][1], matches[2][1])


class TestAltNameHandling(unittest.TestCase):
    """Test alt name cleaning and extraction."""

    def test_clean_alt_name_normal(self):
        """Test cleaning normal alt name."""
        self.assertEqual(clean_alt_name("  Solo Leveling  "), "Solo Leveling")
        self.assertEqual(clean_alt_name("Tower of God"), "Tower of God")

    def test_clean_alt_name_none(self):
        """Test cleaning None returns UNKNOWN."""
        self.assertEqual(clean_alt_name(None), UNKNOWN)

    def test_clean_alt_name_empty(self):
        """Test cleaning empty string returns UNKNOWN."""
        self.assertEqual(clean_alt_name(""), UNKNOWN)

    def test_split_alt_names(self):
        """Test splitting comma-separated alt names."""
        df = pd.DataFrame({"altName": ["Name1, Name2, Name3", "AltA, AltB"]})
        result = split_alt_names(df)
        self.assertEqual(result.iloc[0, 0], "Name1")
        self.assertEqual(result.iloc[0, 1], " Name2")
        self.assertEqual(result.iloc[0, 2], " Name3")

    def test_map_alt_names_to_list(self):
        """Test mapping alt names to a list."""
        df = pd.DataFrame(
            {"altName": ["Name1, Name2, Name3", "AltA, AltB", "Single"]}
        )
        result = map_alt_names_to_list(df)
        # Should contain all names, cleaned
        self.assertIn("Name1", result)
        self.assertIn("Name2", result)
        self.assertIn("Name3", result)
        self.assertIn("AltA", result)
        self.assertIn("AltB", result)
        self.assertIn("Single", result)

    def test_map_alt_names_to_list_filters_unknown(self):
        """Test that UNKNOWN values are filtered out."""
        df = pd.DataFrame({"altName": ["Name1, Name2", "", None]})
        result = map_alt_names_to_list(df)
        self.assertNotIn(UNKNOWN, result)


class TestFindManwhaByName(unittest.TestCase):
    """Test manwha finding functionality."""

    def test_find_manwha_exact_match(self):
        """Test finding manwha with exact name match."""
        df = pd.DataFrame(
            {
                "name": ["Solo Leveling", "Tower of God", "The Beginning After The End"],
                "rating": [4.8, 4.5, 4.7],
                "altName": ["Na Honjaman Level-Up", "Sinui Tap", "TBATE"],
            }
        )
        result = find_manwha_by_name(df, "Solo Leveling")
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result["name"], "Solo Leveling")
        self.assertEqual(result["rating"], 4.8)

    def test_find_manwha_fuzzy_match(self):
        """Test finding manwha with fuzzy match."""
        df = pd.DataFrame(
            {
                "name": ["Solo Leveling", "Tower of God"],
                "rating": [4.8, 4.5],
                "altName": ["Alt1", "Alt2"],
            }
        )
        result = find_manwha_by_name(df, "Solo Levling")  # Typo
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result["name"], "Solo Leveling")

    def test_find_manwha_not_found(self):
        """Test None is returned when manwha not found."""
        df = pd.DataFrame(
            {"name": ["Solo Leveling", "Tower of God"], "rating": [4.8, 4.5], "altName": ["", ""]}
        )
        result = find_manwha_by_name(df, "Completely Different Nonexistent Title xyz123abc")
        # Should return None when nothing matches above threshold (as per function docstring)
        self.assertIsNone(result)

    def test_find_manwha_by_alt_name(self):
        """Test finding manwha by alt name."""
        df = pd.DataFrame(
            {
                "name": ["Solo Leveling", "Tower of God"],
                "rating": [4.8, 4.5],
                "altName": ["Na Honjaman Level-Up", "Sinui Tap"],
            }
        )
        result = find_manwha_by_name(df, "Na Honjaman Level-Up")
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result["name"], "Solo Leveling")


class TestGetTargetManwha(unittest.TestCase):
    """Test fuzzy matching for manwha names."""

    def test_exact_match_found(self):
        """Test exact name match."""
        df = pd.DataFrame(
            {
                "name": ["Solo Leveling", "Tower of God", "The Beginning After The End"],
                "altName": ["Alt1", "Alt2", "Alt3"],
            }
        )
        idx, name = get_target_manwha(df, "Solo Leveling")
        self.assertEqual(name, "Solo Leveling")
        # idx can be int or np.integer (numpy integer types are subclasses of np.integer)
        self.assertTrue(isinstance(idx, (int, np.integer)))

    def test_fuzzy_match_above_threshold(self):
        """Test fuzzy match with typo above 0.70 threshold."""
        df = pd.DataFrame(
            {
                "name": ["Solo Leveling", "Tower of God"],
                "altName": ["Alt1", "Alt2"],
            }
        )
        idx, name = get_target_manwha(df, "Solo Levling")  # Typo
        self.assertEqual(name, "Solo Leveling")

    def test_alt_name_match(self):
        """Test matching by alt name."""
        df = pd.DataFrame(
            {
                "name": ["Solo Leveling", "Tower of God"],
                "altName": ["Na Honjaman Level-Up, Only I Level Up", "Sinui Tap"],
            }
        )
        idx, name = get_target_manwha(df, "Na Honjaman Level-Up")
        self.assertEqual(name, "Solo Leveling")

    def test_alt_name_fuzzy_match(self):
        """Test fuzzy matching on alt names."""
        df = pd.DataFrame(
            {
                "name": ["Solo Leveling", "Tower of God"],
                "altName": ["Na Honjaman Level-Up", "Sinui Tap"],
            }
        )
        # Fuzzy match on alt name
        idx, name = get_target_manwha(df, "Honjaman Level")
        self.assertEqual(name, "Solo Leveling")

    def test_no_match_raises_error(self):
        """Test error raised when no match found."""
        df = pd.DataFrame(
            {"name": ["Solo Leveling", "Tower of God"], "altName": ["Alt1", "Alt2"]}
        )
        with self.assertRaises(IndexError) as context:
            get_target_manwha(df, "Completely Different Unrelated Title xyz123")
        self.assertIn("not found", str(context.exception))

    def test_prefers_higher_similarity(self):
        """Test that higher similarity match is preferred."""
        df = pd.DataFrame(
            {
                "name": ["Solo Leveling", "Solo Player", "Tower of God"],
                "altName": ["", "", ""],
            }
        )
        idx, name = get_target_manwha(df, "Solo Leveling 2")
        # Should match "Solo Leveling" as it's more similar
        self.assertEqual(name, "Solo Leveling")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test behavior with empty dataframe."""
        df = pd.DataFrame({"name": [], "altName": []})
        with self.assertRaises(IndexError):
            get_target_manwha(df, "Any Title")

    def test_single_entry_dataframe(self):
        """Test with single entry dataframe."""
        df = pd.DataFrame({"name": ["Solo Leveling"], "altName": ["Alt1"]})
        idx, name = get_target_manwha(df, "Solo Leveling")
        self.assertEqual(name, "Solo Leveling")

    def test_special_characters_in_names(self):
        """Test matching with special characters."""
        df = pd.DataFrame(
            {
                "name": ["Re:Zero", "Sword Art Online", "No Game No Life"],
                "altName": ["", "", ""],
            }
        )
        idx, name = get_target_manwha(df, "Re:Zero")
        self.assertEqual(name, "Re:Zero")

    def test_numbers_in_names(self):
        """Test matching with numbers in names."""
        df = pd.DataFrame(
            {"name": ["Solo Leveling 2", "Tower of God"], "altName": ["", ""]}
        )
        idx, name = get_target_manwha(df, "Solo Leveling 2")
        self.assertEqual(name, "Solo Leveling 2")

    def test_very_long_names(self):
        """Test matching with very long names."""
        long_name = "The Beginning After The End: A Very Long Title With Many Words"
        df = pd.DataFrame({"name": [long_name, "Short"], "altName": ["", ""]})
        idx, name = get_target_manwha(df, long_name)  # Exact match to avoid edge case
        self.assertEqual(name, long_name)

    def test_find_manwha_returns_series_when_found(self):
        """Test Series is returned when manwha found."""
        df = pd.DataFrame(
            {"name": ["Solo Leveling", "Tower of God"], "rating": [4.5, 4.0], "altName": ["", ""]}
        )
        result = find_manwha_by_name(df, "Solo Leveling")
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result["name"], "Solo Leveling")


if __name__ == "__main__":
    unittest.main()
