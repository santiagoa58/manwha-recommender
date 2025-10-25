"""Tests for parse_manwha.py module."""

import unittest
from scripts.parse_manwha import (
    clean_text,
    unidecode,
    is_str_number,
    parse_number,
    clean_value,
)
from src.utils.constants import UNKNOWN


class TestTextCleaning(unittest.TestCase):
    """Test text cleaning functionality."""

    def test_clean_text_removes_nonprintable(self):
        """Test non-printable characters are removed."""
        result = clean_text("Hello\x00World\x01Test\x02")
        # Non-printable chars should be removed
        self.assertNotIn("\x00", result)
        self.assertNotIn("\x01", result)
        self.assertNotIn("\x02", result)

    def test_clean_text_keeps_printable(self):
        """Test printable ASCII is kept."""
        result = clean_text("Hello World 123!@#")
        self.assertEqual(result, "Hello World 123!@#")

    def test_clean_text_handles_none(self):
        """Test None input returns UNKNOWN."""
        self.assertEqual(clean_text(None), UNKNOWN)

    def test_clean_text_handles_empty(self):
        """Test empty string returns UNKNOWN."""
        self.assertEqual(clean_text(""), UNKNOWN)

    def test_clean_text_handles_whitespace_only(self):
        """Test whitespace-only string."""
        result = clean_text("   ")
        self.assertIsInstance(result, str)

    def test_unidecode_basic(self):
        """Test unidecode converts non-ASCII to ASCII range."""
        result = unidecode("Hello World")
        self.assertEqual(result, "Hello World")

    def test_unidecode_removes_non_ascii(self):
        """Test unidecode removes characters outside ASCII printable range."""
        result = unidecode("Café")
        # Characters outside ASCII 32-126 should be removed
        self.assertNotIn("é", result)

    def test_unidecode_keeps_ascii_printable(self):
        """Test unidecode keeps printable ASCII."""
        text = "Hello World 123!@#$%"
        result = unidecode(text)
        self.assertEqual(result, text)


class TestNumberParsing(unittest.TestCase):
    """Test number parsing from strings."""

    def test_parse_number_valid_integer(self):
        """Test parsing valid integer strings."""
        self.assertEqual(parse_number("123"), 123)
        self.assertEqual(parse_number("0"), 0)
        self.assertEqual(parse_number("-42"), -42)

    def test_parse_number_valid_float(self):
        """Test parsing valid float strings."""
        self.assertEqual(parse_number("12.34"), 12.34)
        self.assertEqual(parse_number("0.5"), 0.5)
        self.assertEqual(parse_number("-3.14"), -3.14)

    def test_parse_number_prefers_int(self):
        """Test integer strings return int, not float."""
        result = parse_number("42")
        self.assertIsInstance(result, int)
        self.assertEqual(result, 42)

    def test_is_str_number_valid_integers(self):
        """Test valid integer strings."""
        self.assertTrue(is_str_number("123"))
        self.assertTrue(is_str_number("0"))
        self.assertTrue(is_str_number("-999"))

    def test_is_str_number_valid_floats(self):
        """Test valid float strings."""
        self.assertTrue(is_str_number("12.34"))
        self.assertTrue(is_str_number("0.5"))
        self.assertTrue(is_str_number("-3.14"))

    def test_is_str_number_invalid(self):
        """Test invalid number strings."""
        self.assertFalse(is_str_number("abc"))
        self.assertFalse(is_str_number("12.34.56"))
        self.assertFalse(is_str_number(""))
        self.assertFalse(is_str_number("not a number"))


class TestCleanValue(unittest.TestCase):
    """Test clean_value function for mixed type cleaning."""

    def test_clean_value_integer(self):
        """Test integer values pass through unchanged."""
        self.assertEqual(clean_value(42), 42)
        self.assertEqual(clean_value(0), 0)
        self.assertEqual(clean_value(-10), -10)

    def test_clean_value_float(self):
        """Test float values pass through unchanged."""
        self.assertEqual(clean_value(3.14), 3.14)
        self.assertEqual(clean_value(0.0), 0.0)

    def test_clean_value_none(self):
        """Test None returns UNKNOWN."""
        self.assertEqual(clean_value(None), UNKNOWN)

    def test_clean_value_empty_string(self):
        """Test empty string returns UNKNOWN."""
        self.assertEqual(clean_value(""), UNKNOWN)

    def test_clean_value_numeric_string(self):
        """Test numeric string gets converted to number."""
        self.assertEqual(clean_value("123"), 123)
        self.assertEqual(clean_value("45.67"), 45.67)

    def test_clean_value_text_string(self):
        """Test text string gets cleaned."""
        result = clean_value("Hello World")
        self.assertEqual(result, "Hello World")

    def test_clean_value_zero_not_unknown(self):
        """Test zero is not treated as falsy/UNKNOWN."""
        self.assertEqual(clean_value(0), 0)
        self.assertEqual(clean_value(0.0), 0.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_clean_text_unicode_extended(self):
        """Test handling of extended Unicode characters."""
        result = clean_text("Hello™")
        # Extended characters should be filtered
        self.assertIsInstance(result, str)

    def test_clean_text_mixed_printable_nonprintable(self):
        """Test mixed printable and non-printable characters."""
        result = clean_text("Hello\nWorld\tTest")
        # Newlines and tabs should be handled
        self.assertIsInstance(result, str)

    def test_parse_number_edge_values(self):
        """Test edge values for number parsing."""
        self.assertEqual(parse_number("0"), 0)
        self.assertEqual(parse_number("0.0"), 0.0)

    def test_is_str_number_edge_cases(self):
        """Test edge cases for number detection."""
        self.assertTrue(is_str_number("0"))
        self.assertTrue(is_str_number("0.0"))
        self.assertFalse(is_str_number(" "))

    def test_clean_value_false_passes_through(self):
        """Test that boolean False passes through as-is."""
        # Note: In Python, False == 0, so `value != 0` is False for False
        # Also, bool is a subclass of int, so isinstance(False, int) is True
        # This test documents the current behavior
        result = clean_value(False)
        self.assertEqual(result, False)

    def test_clean_value_whitespace_string(self):
        """Test whitespace-only string."""
        result = clean_value("   ")
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
