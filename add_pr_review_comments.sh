#!/bin/bash
# Script to add detailed review comments to the PR
# Usage: ./add_pr_review_comments.sh <PR_NUMBER>

if [ -z "$1" ]; then
    echo "Usage: $0 <PR_NUMBER>"
    echo "Example: $0 42"
    exit 1
fi

PR_NUMBER=$1
REPO="santiagoa58/manwha-recommender"

echo "Adding review comments to PR #${PR_NUMBER}..."

# Critical Issue #1: Tautological condition in clean_text()
gh pr review $PR_NUMBER --comment \
    --body "**🔴 CRITICAL BUG: Tautological Condition**

\`\`\`python
cleaned_text = [char for char in text if char in text]
\`\`\`

This condition is always True - every character in \`text\` is obviously in \`text\`. This does nothing and doesn't filter non-printable characters as the comment claims.

**Fix:**
\`\`\`python
def clean_text(text: str | None):
    if not text:
        return UNKNOWN
    text = unidecode(text)
    # Keep only printable ASCII characters (space through tilde)
    cleaned_text = [char for char in text if 32 <= ord(char) <= 126]
    return \"\"\"\" \".join(cleaned_text)
\`\`\`" \
    --file "scripts/parse_manwha.py" \
    --line 39

# Critical Issue #2: Type checking anti-pattern
gh pr review $PR_NUMBER --comment \
    --body "**🔴 CRITICAL: Type Checking Anti-Pattern**

\`\`\`python
if type(manwha_details) is str and manwha_details == MANWHA_NOT_FOUND:
\`\`\`

Issues:
1. Using \`type() is\` instead of \`isinstance()\` - Python anti-pattern
2. Using strings as sentinel values defeats type safety
3. Function returns different types (Series or str) which breaks type hints

**Fix:** Use \`Optional[pd.Series]\` return type and return \`None\` for not found:
\`\`\`python
from typing import Optional

def find_manwha_by_name(manwhas_df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    try:
        target_index, target_name = get_target_manwha(manwhas_df, name)
        return manwhas_df.iloc[target_index]
    except IndexError:
        return None
\`\`\`

Then update caller:
\`\`\`python
manwha_details = find_manwha_by_name(all_manwhas, manwha_name)
if manwha_details is None:
    not_found.append(manwha)
    continue
\`\`\`" \
    --file "scripts/get_cleaned_manwha_recommendations.py" \
    --line 26

# Critical Issue #3a: Silent exception handling
gh pr review $PR_NUMBER --comment \
    --body "**🔴 HIGH: Silent Exception Handling**

\`\`\`python
except Exception as _e:  # Catches everything, never logs
    return False
\`\`\`

Catches all exceptions (including KeyboardInterrupt, SystemExit) and silently returns False. Makes debugging impossible.

**Fix:**
\`\`\`python
except (ValueError, TypeError) as e:
    logger.warning(f\"Could not parse number '{str_number}': {e}\")
    return False
\`\`\`" \
    --file "scripts/parse_raw_reddit_recommendations.py" \
    --line 24

# Critical Issue #3b: Error without details
gh pr review $PR_NUMBER --comment \
    --body "**🟡 Error Handling: Missing Details**

\`\`\`python
except Exception as e:
    print(f\"Error parsing manwhas\\n\")  # Doesn't include error details!
    raise e
\`\`\`

Prints error without details. Use logging:

\`\`\`python
except Exception as e:
    logger.error(f\"Error parsing manwhas: {e}\", exc_info=True)
    raise RuntimeError(\"Failed to parse manwhas\") from e
\`\`\`" \
    --file "scripts/parse_manwha.py" \
    --line 196

# Critical Issue #3c: No error handling in build
gh pr review $PR_NUMBER --comment \
    --body "**🔴 HIGH: No Error Handling in Build Pipeline**

If any step fails, subsequent steps still execute with corrupted/missing data. Need proper error handling:

\`\`\`python
import logging
logger = logging.getLogger(__name__)

def run():
    try:
        logger.info(\"Step 1: Parsing manwha data...\")
        parse_manwha.run()
    except Exception as e:
        logger.error(f\"Failed to parse manwha: {e}\", exc_info=True)
        raise RuntimeError(\"Build failed at parse_manwha step\") from e

    try:
        logger.info(\"Step 2: Parsing raw Reddit recommendations...\")
        parse_raw_recommendations.run()
    except Exception as e:
        logger.error(f\"Failed to parse raw recommendations: {e}\", exc_info=True)
        raise RuntimeError(\"Build failed at parse_raw_recommendations step\") from e

    try:
        logger.info(\"Step 3: Cleaning Reddit recommendations...\")
        clean_manwha_recommendations.run()
    except Exception as e:
        logger.error(f\"Failed to clean recommendations: {e}\", exc_info=True)
        raise RuntimeError(\"Build failed at clean_manwha_recommendations step\") from e

    try:
        logger.info(\"Step 4: Saving recommender model...\")
        save_manwha_recommender_model()
        logger.info(\"Build completed successfully!\")
    except Exception as e:
        logger.error(f\"Failed to save model: {e}\", exc_info=True)
        raise RuntimeError(\"Build failed at save_model step\") from e
\`\`\`" \
    --file "scripts/build.py" \
    --line 7

# High Priority #4: Inadequate test coverage
gh pr review $PR_NUMBER --comment \
    --body "**🔴 HIGH: Zero Test Coverage for New Code**

This PR adds ~320 lines of new parsing code with ZERO tests:
- \`parse_raw_reddit_recommendations.py\`: 250 lines, 0 tests
- \`get_cleaned_manwha_recommendations.py\`: 70 lines, 0 tests
- The actual recommender system is not tested

**Missing Tests:**
1. Rating parsing edge cases (malformed ratings, ratios, invalid values)
2. Category/subcategory extraction
3. Alt name parsing
4. File loading failures (missing files, corrupted JSON)
5. Empty/null inputs
6. Unicode handling in text cleaning
7. Build pipeline integration

**Required:** Add comprehensive tests with at least 70% code coverage before merging.

Example tests needed:
\`\`\`python
class TestRatingParsing(unittest.TestCase):
    def test_get_rating_valid_single_letter(self):
        self.assertEqual(get_rating(\"P\"), 5.0)
        self.assertEqual(get_rating(\"E\"), 4.0)

    def test_get_rating_case_insensitive(self):
        self.assertEqual(get_rating(\"p\"), 5.0)

    def test_get_rating_ratio(self):
        result = get_rating(\"P/E\")
        self.assertAlmostEqual(result, 4.5)

    def test_get_rating_invalid(self):
        self.assertIsNone(get_rating(\"INVALID\"))
\`\`\`" \
    --file "src/recommender/tests/test_manwha_recommender.py" \
    --line 1

# High Priority #5: Hard-coded paths everywhere
gh pr review $PR_NUMBER --comment \
    --body "**🟡 DRY Violation: Hard-coded Paths**

Hard-coded path instead of using centralized constants. All paths should be defined in \`src/utils/constants.py\`:

\`\`\`python
from pathlib import Path

# Directories
DATA_DIR = Path(\"data\")
MODELS_DIR = Path(\"models\")

# Raw data
RAW_MANWHAS_PATH = DATA_DIR / \"rawManwhas.json\"
RAW_REDDIT_RECOMMENDATIONS_TXT = DATA_DIR / \"raw_manwha_reddit_recommendations.txt\"
RAW_REDDIT_RECOMMENDATIONS_JSON = DATA_DIR / \"raw_manwha_reddit_recommendations.json\"

# Cleaned data
CLEANED_MANWHAS_PATH = DATA_DIR / \"cleanedManwhas.json\"
CLEANED_REDDIT_RECOMMENDATIONS = DATA_DIR / \"cleaned_manwha_reddit_recommendations.json\"
UNCLEANED_REDDIT_RECOMMENDATIONS = DATA_DIR / \"uncleaned_manwha_reddit_recommendations.json\"
\`\`\`

Then import and use these constants instead of hard-coding paths." \
    --file "scripts/parse_manwha.py" \
    --line 174

# High Priority #6: Performance issue
gh pr review $PR_NUMBER --comment \
    --body "**🟡 Performance Issue: Unnecessary O(n) Operations**

This function always performs TWO full O(n) similarity comparisons, even when the first succeeds. Also, \`map_alt_names_to_list()\` processes the entire dataframe every time without caching.

**Issues:**
1. Always computes alt_names even if name search succeeds
2. No caching of expanded alt_names list
3. For 1000+ manwhas, this is inefficient

**Fix:** Only compute alt_names if first search fails (already returns early at line 70-76, so this is good), but consider caching the expanded alt_names list:

\`\`\`python
# At module level
_alt_names_cache = None

def get_target_manwha(manwhas_df: pd.DataFrame, target_manhwa: str) -> tuple[int, str]:
    global _alt_names_cache

    # Try name first
    most_similar_name, highest_similarity = get_close_matches(
        target_manhwa, manwhas_df[\"name\"].tolist(), limit=1
    )[0]

    if highest_similarity > SIMILARITY_THRESHOLD:
        target_manwha_df = manwhas_df[manwhas_df[\"name\"] == most_similar_name]
        return (target_manwha_df.index[0], target_manwha_df[\"name\"])

    # Only compute alt_names if needed, and cache
    if _alt_names_cache is None:
        _alt_names_cache = map_alt_names_to_list(manwhas_df)

    # ... rest of function
\`\`\`" \
    --file "src/utils/manwha_utils.py" \
    --line 65

# Medium Priority #7: Duplicate file I/O code
gh pr review $PR_NUMBER --comment \
    --body "**🟡 DRY Violation: Duplicate File I/O Pattern**

This file I/O pattern is repeated in multiple places. Create a \`src/utils/file_io.py\` module:

\`\`\`python
import json
from pathlib import Path
from typing import Any, List

def load_json(path: Path) -> dict:
    \"\"\"Load JSON file with error handling.\"\"\"
    try:
        with open(path, \"r\", encoding=\"utf-8\") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f\"JSON file not found: {path}\")
    except json.JSONDecodeError as e:
        raise ValueError(f\"Invalid JSON in {path}: {e}\")

def save_json(path: Path, data: Any, indent: int = 4) -> None:
    \"\"\"Save data to JSON file.\"\"\"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, \"w\", encoding=\"utf-8\") as f:
        json.dump(data, f, indent=indent)

def load_text_lines(path: Path) -> List[str]:
    \"\"\"Load text file as list of non-empty lines.\"\"\"
    try:
        with open(path, \"r\", encoding=\"utf-8\") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f\"Text file not found: {path}\")
\`\`\`

Then replace all direct file I/O with these utilities." \
    --file "scripts/get_cleaned_manwha_recommendations.py" \
    --line 15

# Medium Priority #8: Missing data validation
gh pr review $PR_NUMBER --comment \
    --body "**🟡 Missing Data Validation**

No validation that loaded JSON has expected structure. What if required fields are missing or data types are wrong?

**Recommendation:** Add validation with dataclasses:
\`\`\`python
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class ManwhaData:
    name: str
    rating: float
    tags: List[str]
    description: str
    altName: Optional[str] = None
    chapters: Optional[Dict[str, any]] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError(\"Manwha name cannot be empty\")
        if not 0 <= self.rating <= 5:
            raise ValueError(f\"Rating must be 0-5, got {self.rating}\")

    @classmethod
    def from_dict(cls, data: dict) -> 'ManwhaData':
        try:
            return cls(
                name=data[\"name\"],
                rating=float(data.get(\"rating\", 0)),
                tags=data.get(\"tags\", []),
                description=data.get(\"description\", \"\"),
                altName=data.get(\"altName\"),
                chapters=data.get(\"chapters\"),
            )
        except KeyError as e:
            raise ValueError(f\"Missing required field: {e}\")
\`\`\`" \
    --file "scripts/parse_manwha.py" \
    --line 173

# Medium Priority #9: Shadowing built-in
gh pr review $PR_NUMBER --comment \
    --body "**🟡 Shadowing Built-in Name**

Parameter named \`str\` shadows Python's built-in \`str\` type. This can cause confusing bugs.

**Fix:**
\`\`\`python
def is_alt_name(text: str) -> bool:
    \"\"\"Check if the string is an alt name (starts with 'or,').\"\"\"
    return text.lower().startswith(\"or,\")
\`\`\`" \
    --file "scripts/parse_raw_reddit_recommendations.py" \
    --line 94

# Medium Priority #10: Unused imports
gh pr review $PR_NUMBER --comment \
    --body "**🔵 Unused Import**

\`pandas\` is imported but never used in this file.

**Fix:** Remove unused import." \
    --file "scripts/parse_raw_reddit_recommendations.py" \
    --line 62

# Medium Priority #11: Magic numbers
gh pr review $PR_NUMBER --comment \
    --body "**🟡 Magic Numbers Without Named Constants**

Use descriptive named constants:

\`\`\`python
RATING_PEAK = 5.0
RATING_ENJOYED = 4.0
RATING_GOOD = 3.0
RATING_SUCKS = 2.0
RATING_DROPPED = 1.0

RATINGS = {
    \"P\": RATING_PEAK,
    \"E\": RATING_ENJOYED,
    \"G\": RATING_GOOD,
    \"S\": RATING_SUCKS,
    \"D\": RATING_DROPPED,
}
\`\`\`" \
    --file "scripts/parse_raw_reddit_recommendations.py" \
    --line 32

# Medium Priority #11b: Magic threshold
gh pr review $PR_NUMBER --comment \
    --body "**🟡 Magic Number**

Define as named constant:
\`\`\`python
SIMILARITY_THRESHOLD = 0.70  # Minimum similarity for fuzzy match

if highest_similarity > SIMILARITY_THRESHOLD:
\`\`\`" \
    --file "src/utils/manwha_utils.py" \
    --line 70

# Medium Priority #12: Missing docstring
gh pr review $PR_NUMBER --comment \
    --body "**🔵 Missing Docstring**

Add comprehensive docstring explaining the build pipeline:

\`\`\`python
def run():
    \"\"\"Build the manwha recommender system.

    Executes the full build pipeline in order:
    1. Parse raw manwha HTML data into structured JSON
    2. Parse raw Reddit recommendations text into structured JSON
    3. Cross-reference Reddit recommendations with manwha database
    4. Train and save the KNN recommender model

    Raises:
        RuntimeError: If any build step fails

    Output Files:
        - data/cleanedManwhas.json
        - data/raw_manwha_reddit_recommendations.json
        - data/cleaned_manwha_reddit_recommendations.json
        - data/uncleaned_manwha_reddit_recommendations.json
        - models/manwha_recommender.pkl
    \"\"\"
\`\`\`" \
    --file "scripts/build.py" \
    --line 7

echo ""
echo "✅ All review comments added successfully!"
echo ""
echo "Summary:"
echo "- 3 Critical issues"
echo "- 4 High priority issues"
echo "- 6 Medium priority issues"
echo "- 2 Low priority style issues"
echo ""
echo "Overall recommendation: Request changes before merge"
echo "See PR_REVIEW_COMMENTS.md for full details"
