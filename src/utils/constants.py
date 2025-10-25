from pathlib import Path

# Directories
DATA_DIR = Path("data")
MODELS_DIR = Path("models")

# Raw data
RAW_MANWHAS_PATH = DATA_DIR / "rawManwhas.json"
RAW_REDDIT_RECOMMENDATIONS_TXT = DATA_DIR / "raw_manwha_reddit_recommendations.txt"
RAW_REDDIT_RECOMMENDATIONS_JSON = DATA_DIR / "raw_manwha_reddit_recommendations.json"

# Cleaned data
CLEANED_MANWHAS_PATH = DATA_DIR / "cleanedManwhas.json"
CLEANED_REDDIT_RECOMMENDATIONS = DATA_DIR / "cleaned_manwha_reddit_recommendations.json"
UNCLEANED_REDDIT_RECOMMENDATIONS = DATA_DIR / "uncleaned_manwha_reddit_recommendations.json"

# Models
MANWHA_RECOMMENDER_MODEL_PATH = MODELS_DIR / "manwha_recommender.pkl"

# Sentinel values
UNKNOWN = "Unknown"
MANWHA_NOT_FOUND = "The manhwa you entered is not in our database."
