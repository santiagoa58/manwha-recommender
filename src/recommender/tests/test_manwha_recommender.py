import unittest
from src.utils.manwha_utils import find_manwha_by_name, get_manwhas
from src.utils.constants import MANWHA_NOT_FOUND


class TestManwhaRecommender(unittest.TestCase):
    def test_find_manwha_by_name(self):
        manwhas_df = get_manwhas()
        # manwha found with name
        manwha = find_manwha_by_name(manwhas_df, "teenage Mercenary")
        self.assertEqual(manwha["name"], "Teenage Mercenary")
        # manwha found with alt name
        manwha = find_manwha_by_name(manwhas_df, "Ibeon Saengeun Gajuga Doegetseumnida")
        self.assertEqual(manwha["name"], "I Shall Master This Family")
        # manwha not found
        manwha_not_found = find_manwha_by_name(manwhas_df, "asbasdfasfsfsf")
        self.assertEqual(manwha_not_found, MANWHA_NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
