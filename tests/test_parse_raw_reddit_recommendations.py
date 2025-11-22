"""Tests for parse_raw_reddit_recommendations.py module."""

import unittest
from scripts.parse_raw_reddit_recommendations import (
    get_rating,
    get_rating_from_title,
    get_rating_ratio,
    is_alt_name,
    is_category,
    is_subcategory,
    get_subcategory,
    get_notes_from_title,
    get_alt_names_from_notes,
    RATINGS,
)


class TestRatingParsing(unittest.TestCase):
    """Test rating parsing from various formats."""

    def test_get_rating_valid_single_letter(self):
        """Test single letter ratings."""
        self.assertEqual(get_rating("P"), RATINGS["P"])
        self.assertEqual(get_rating("E"), RATINGS["E"])
        self.assertEqual(get_rating("G"), RATINGS["G"])
        self.assertEqual(get_rating("S"), RATINGS["S"])
        self.assertEqual(get_rating("D"), RATINGS["D"])

    def test_get_rating_case_insensitive(self):
        """Test ratings are case-insensitive."""
        self.assertEqual(get_rating("p"), RATINGS["P"])
        self.assertEqual(get_rating("Peak"), RATINGS["P"])
        self.assertEqual(get_rating("ENJOYED"), RATINGS["E"])
        self.assertEqual(get_rating("good"), RATINGS["G"])

    def test_get_rating_full_words(self):
        """Test full word ratings."""
        self.assertEqual(get_rating("Best"), RATINGS["P"])
        self.assertEqual(get_rating("Enjoyed Reading It"), RATINGS["E"])
        self.assertEqual(get_rating("Good Enough"), RATINGS["G"])
        self.assertEqual(get_rating("Sucks"), RATINGS["S"])
        self.assertEqual(get_rating("Dropped"), RATINGS["D"])

    def test_get_rating_ratio(self):
        """Test ratio ratings like P/E."""
        result = get_rating("P/E")
        expected = (RATINGS["P"] + RATINGS["E"]) / 2
        self.assertAlmostEqual(result, expected)

    def test_get_rating_ratio_multiple(self):
        """Test ratio with multiple ratings."""
        result = get_rating("P/E/G")
        expected = (RATINGS["P"] + RATINGS["E"] + RATINGS["G"]) / 3
        self.assertAlmostEqual(result, expected)

    def test_get_rating_invalid(self):
        """Test invalid ratings return None."""
        self.assertIsNone(get_rating("INVALID"))
        self.assertIsNone(get_rating(""))
        self.assertIsNone(get_rating("XYZ"))
        self.assertIsNone(get_rating("123"))

    def test_get_rating_from_title_single_rating(self):
        """Test extracting rating from title with parentheses."""
        self.assertEqual(get_rating_from_title("Solo Leveling (P)"), RATINGS["P"])
        self.assertEqual(get_rating_from_title("Tower of God (E)"), RATINGS["E"])

    def test_get_rating_from_title_with_notes(self):
        """Test extracting rating from title with notes."""
        self.assertEqual(
            get_rating_from_title("Return of Mount Hua Sect (comedy) (E)"), RATINGS["E"]
        )

    def test_get_rating_from_title_no_rating(self):
        """Test title with no rating returns None."""
        self.assertIsNone(get_rating_from_title("No Rating Here"))
        self.assertIsNone(get_rating_from_title("Just a Title (comedy)"))

    def test_get_rating_ratio_function(self):
        """Test get_rating_ratio function directly."""
        result = get_rating_ratio("P/E")
        expected = (RATINGS["P"] + RATINGS["E"]) / 2
        self.assertAlmostEqual(result, expected)

    def test_get_rating_ratio_with_invalid(self):
        """Test ratio with invalid rating returns average of valid ones."""
        result = get_rating_ratio("P/INVALID/E")
        expected = (RATINGS["P"] + RATINGS["E"]) / 2
        self.assertAlmostEqual(result, expected)


class TestNotesExtraction(unittest.TestCase):
    """Test notes extraction from titles."""

    def test_get_notes_from_title_single_note(self):
        """Test extracting single note from title."""
        notes = get_notes_from_title("Solo Leveling (comedy)")
        self.assertEqual(notes, ["comedy"])

    def test_get_notes_from_title_multiple_notes(self):
        """Test extracting multiple notes from title."""
        notes = get_notes_from_title("Tower of God (comedy) (action)")
        self.assertIn("comedy", notes)
        self.assertIn("action", notes)

    def test_get_notes_from_title_with_rating(self):
        """Test notes exclude ratings."""
        notes = get_notes_from_title("Solo Leveling (comedy) (P)")
        self.assertEqual(notes, ["comedy"])

    def test_get_notes_from_title_no_notes(self):
        """Test title with no notes returns None."""
        self.assertIsNone(get_notes_from_title("Solo Leveling"))
        self.assertIsNone(get_notes_from_title("Solo Leveling (P)"))


class TestAltNameDetection(unittest.TestCase):
    """Test alt name detection and extraction."""

    def test_is_alt_name_valid(self):
        """Test strings starting with 'or,'."""
        self.assertTrue(is_alt_name("or, Alternative Title"))
        self.assertTrue(is_alt_name("OR, UPPERCASE"))
        self.assertTrue(is_alt_name("Or, Mixed Case"))

    def test_is_alt_name_invalid(self):
        """Test strings not starting with 'or,'."""
        self.assertFalse(is_alt_name("Regular Title"))
        self.assertFalse(is_alt_name("or something"))
        self.assertFalse(is_alt_name(" or, with space"))

    def test_get_alt_names_from_notes(self):
        """Test extracting alt names from notes."""
        notes = ["comedy", "or, Alt Name 1, Alt Name 2"]
        alt_names = get_alt_names_from_notes(notes)
        self.assertIn("Alt Name 1", alt_names)
        self.assertIn("Alt Name 2", alt_names)

    def test_get_alt_names_from_notes_no_alt(self):
        """Test notes without alt names return None."""
        notes = ["comedy", "action"]
        self.assertIsNone(get_alt_names_from_notes(notes))


class TestCategoryDetection(unittest.TestCase):
    """Test category and subcategory detection."""

    def test_is_category_uppercase(self):
        """Test uppercase strings are categories."""
        self.assertTrue(is_category("ACTION"))
        self.assertTrue(is_category("FANTASY"))
        self.assertTrue(is_category("ROMANCE"))

    def test_is_category_with_lowercase_s(self):
        """Test categories with 's in lowercase."""
        self.assertTrue(is_category("AUTHOR'S PICKS"))

    def test_is_category_mixed_case(self):
        """Test mixed case is not category."""
        self.assertFalse(is_category("Action"))
        self.assertFalse(is_category("Fantasy"))

    def test_is_subcategory_valid(self):
        """Test strings in brackets are subcategories."""
        self.assertTrue(is_subcategory("[Best Action]"))
        self.assertTrue(is_subcategory("[Top Picks]"))

    def test_is_subcategory_invalid(self):
        """Test strings not in brackets are not subcategories."""
        self.assertFalse(is_subcategory("Not a subcategory"))
        self.assertFalse(is_subcategory("[Missing bracket"))
        self.assertFalse(is_subcategory("Missing bracket]"))

    def test_get_subcategory(self):
        """Test extracting subcategory name from brackets."""
        self.assertEqual(get_subcategory("[Best Action]"), "Best Action")
        self.assertEqual(get_subcategory("[Top Picks]"), "Top Picks")

    def test_get_subcategory_with_spaces(self):
        """Test subcategory extraction trims spaces."""
        self.assertEqual(get_subcategory("[ Spaced ]"), "Spaced")

    def test_get_subcategory_invalid(self):
        """Test invalid subcategory returns None."""
        self.assertIsNone(get_subcategory("Not a subcategory"))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_get_rating_whitespace(self):
        """Test rating with extra whitespace."""
        self.assertEqual(get_rating("  P  "), RATINGS["P"])
        self.assertEqual(get_rating("\tE\n"), RATINGS["E"])

    def test_get_rating_ratio_empty_parts(self):
        """Test ratio with empty parts."""
        result = get_rating_ratio("P//E")
        expected = (RATINGS["P"] + RATINGS["E"]) / 2
        self.assertAlmostEqual(result, expected)

    def test_get_rating_ratio_all_invalid(self):
        """Test ratio with all invalid ratings."""
        result = get_rating_ratio("INVALID/BAD")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
