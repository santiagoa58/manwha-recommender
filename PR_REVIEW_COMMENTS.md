# PR Review Comments - Code Quality Issues

## 🔴 CRITICAL ISSUES

### 1. Critical Bug: Tautological Condition in `clean_text()`
**File**: `scripts/parse_manwha.py`
**Line**: 39
**Severity**: CRITICAL

```python
cleaned_text = [char for char in text if char in text]
```

**Issue**: This condition is always True - every character in `text` is obviously in `text`. This list comprehension does nothing and doesn't filter non-printable characters as the comment claims.

**Fix**:
```python
def clean_text(text: str | None):
    if not text:
        return UNKNOWN
    text = unidecode(text)
    # Keep only printable ASCII characters (space through tilde)
    cleaned_text = [char for char in text if 32 <= ord(char) <= 126]
    return "".join(cleaned_text)
```

---

### 2. Type Checking Anti-Pattern - Using Magic Strings for Error Handling
**File**: `scripts/get_cleaned_manwha_recommendations.py`
**Line**: 26
**Severity**: CRITICAL

```python
if type(manwha_details) is str and manwha_details == MANWHA_NOT_FOUND:
```

**Issues**:
1. Using `type() is` instead of `isinstance()` - Python anti-pattern
2. Using strings as sentinel values defeats type safety
3. Forces callers to check for magic strings instead of proper error handling
4. The function returns different types (Series or str) which breaks type hints

**Fix** in `src/utils/manwha_utils.py`:
```python
from typing import Optional

def find_manwha_by_name(manwhas_df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Find a manwha by name or alt name.

    Returns:
        pd.Series if found, None if not found
    """
    try:
        target_index, target_name = get_target_manwha(manwhas_df, name)
        return manwhas_df.iloc[target_index]
    except IndexError:
        return None
```

Then update caller in `scripts/get_cleaned_manwha_recommendations.py`:
```python
manwha_details = find_manwha_by_name(all_manwhas, manwha_name)
if manwha_details is None:
    manwha["title"] = manwha_name
    not_found.append(manwha)
    continue
```

---

### 3. Silent Failures and Poor Error Handling
**File**: `scripts/parse_raw_reddit_recommendations.py`
**Line**: 24
**Severity**: HIGH

```python
except Exception as _e:  # Catches everything, never logs
    return False
```

**Issue**: Catches all exceptions (including KeyboardInterrupt, SystemExit) and silently returns False. The underscore prefix `_e` suggests the error is intentionally ignored. This makes debugging impossible.

**Fix**:
```python
except (ValueError, TypeError) as e:
    logger.warning(f"Could not parse number '{str_number}': {e}")
    return False
```

**File**: `scripts/parse_manwha.py`
**Line**: 196
**Severity**: HIGH

```python
except Exception as e:
    print(f"Error parsing manwhas\n")  # Doesn't include error details!
    raise e
```

**Issue**: Prints error without details, then re-raises. Better to use logging.

**Fix**:
```python
except Exception as e:
    logger.error(f"Error parsing manwhas: {e}", exc_info=True)
    raise RuntimeError("Failed to parse manwhas") from e
```

**File**: `scripts/build.py`
**Lines**: 7-11
**Severity**: HIGH

```python
def run():
    parse_manwha.run()
    parse_raw_recommendations.run()
    clean_manwha_recommendations.run()
    save_manwha_recommender_model()
```

**Issue**: NO error handling at all. If step 1 fails, steps 2-4 still execute with corrupted/missing data.

**Fix**:
```python
import logging
logger = logging.getLogger(__name__)

def run():
    try:
        logger.info("Step 1: Parsing manwha data...")
        parse_manwha.run()
    except Exception as e:
        logger.error(f"Failed to parse manwha: {e}", exc_info=True)
        raise RuntimeError("Build failed at parse_manwha step") from e

    try:
        logger.info("Step 2: Parsing raw Reddit recommendations...")
        parse_raw_recommendations.run()
    except Exception as e:
        logger.error(f"Failed to parse raw recommendations: {e}", exc_info=True)
        raise RuntimeError("Build failed at parse_raw_recommendations step") from e

    try:
        logger.info("Step 3: Cleaning Reddit recommendations...")
        clean_manwha_recommendations.run()
    except Exception as e:
        logger.error(f"Failed to clean recommendations: {e}", exc_info=True)
        raise RuntimeError("Build failed at clean_manwha_recommendations step") from e

    try:
        logger.info("Step 4: Saving recommender model...")
        save_manwha_recommender_model()
        logger.info("Build completed successfully!")
    except Exception as e:
        logger.error(f"Failed to save model: {e}", exc_info=True)
        raise RuntimeError("Build failed at save_model step") from e
```

---

## 🟡 HIGH PRIORITY ISSUES

### 4. Inadequate Test Coverage - ZERO Tests for New Functionality
**File**: `src/recommender/tests/test_manwha_recommender.py`
**Lines**: 1-22
**Severity**: HIGH

**Issue**: This PR adds ~320 lines of new parsing code with ZERO tests:
- `parse_raw_reddit_recommendations.py`: 250 lines, 0 tests
- `get_cleaned_manwha_recommendations.py`: 70 lines, 0 tests
- `parse_manwha.py`: Modified, 0 new tests

The single test file only has 21 lines testing one utility function. Tests depend on real data files (fragile).

**Missing Tests**:
1. Rating parsing edge cases (malformed ratings, ratios, invalid values)
2. Category/subcategory extraction
3. Alt name parsing
4. File loading failures (missing files, corrupted JSON)
5. Empty/null inputs
6. Unicode handling in text cleaning
7. Build pipeline integration
8. The actual recommender system (not tested at all!)

**Required Action**: Add comprehensive tests with at least 70% code coverage before merging.

**Example tests needed**:
```python
# test_parse_raw_reddit_recommendations.py
class TestRatingParsing(unittest.TestCase):
    def test_get_rating_valid_single_letter(self):
        self.assertEqual(get_rating("P"), 5.0)
        self.assertEqual(get_rating("E"), 4.0)

    def test_get_rating_case_insensitive(self):
        self.assertEqual(get_rating("p"), 5.0)
        self.assertEqual(get_rating("Peak"), 5.0)

    def test_get_rating_ratio(self):
        result = get_rating("P/E")
        self.assertAlmostEqual(result, 4.5)

    def test_get_rating_invalid(self):
        self.assertIsNone(get_rating("INVALID"))
        self.assertIsNone(get_rating(""))

    def test_get_rating_from_title(self):
        self.assertEqual(get_rating_from_title("Solo Leveling (P)"), 5.0)
        self.assertEqual(get_rating_from_title("Tower of God (comedy) (E)"), 4.0)

class TestTextCleaning(unittest.TestCase):
    def test_clean_text_removes_nonprintable(self):
        # After fixing the bug!
        result = clean_text("Hello\x00World\x01")
        self.assertEqual(result, "HelloWorld")

    def test_clean_text_handles_none(self):
        self.assertEqual(clean_text(None), UNKNOWN)
```

---

### 5. Hard-coded Paths Everywhere - DRY Violation
**Files**: Multiple
**Severity**: MEDIUM-HIGH

**Violations**:
- `scripts/parse_manwha.py:174`: `"./data/rawManwhas.json"`
- `scripts/get_cleaned_manwha_recommendations.py:15`: `"data/raw_manwha_reddit_recommendations.json"`
- `scripts/get_cleaned_manwha_recommendations.py:6-11`: Defines own path constants
- `scripts/parse_raw_reddit_recommendations.py:40-43`: Defines own path constants
- `src/recommender/manwha_recommender.py:9`: `MODEL_PATH` defined locally
- `src/utils/constants.py:3`: Only has ONE path constant

**Fix**: Centralize ALL paths in `src/utils/constants.py`:
```python
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
```

Then update all files to import from constants.

---

### 6. Performance Issue - Unnecessary O(n) Operations
**File**: `src/utils/manwha_utils.py`
**Lines**: 65-95
**Severity**: MEDIUM-HIGH

```python
def get_target_manwha(manwhas_df: pd.DataFrame, target_manhwa: str) -> tuple[int, str]:
    # First O(n) scan through ALL names
    most_similar_name, highest_similarity = get_close_matches(
        target_manhwa, manwhas_df["name"].tolist(), limit=1
    )[0]

    if highest_similarity > 0.70:
        # Found - return early
        ...

    # Second O(n) scan through ALL alt names (even if first search succeeded!)
    alt_names = map_alt_names_to_list(manwhas_df)  # Processes entire dataframe
    most_similar_alt_name, highest_alt_similarity = get_close_matches(
        target_manhwa, alt_names, limit=1
    )[0]
```

**Issues**:
1. `map_alt_names_to_list()` processes entire dataframe every time (not cached)
2. Performs TWO full O(n) similarity comparisons
3. `get_close_matches()` computes similarity for EVERY name in dataset
4. For 1000+ manwhas, this is inefficient

**Fix**:
```python
def get_target_manwha(manwhas_df: pd.DataFrame, target_manhwa: str) -> tuple[int, str]:
    # First try exact name matches
    most_similar_name, highest_similarity = get_close_matches(
        target_manhwa, manwhas_df["name"].tolist(), limit=1
    )[0]

    if highest_similarity > 0.70:
        target_manwha_df = manwhas_df[manwhas_df["name"] == most_similar_name]
        return (target_manwha_df.index[0], target_manwha_df["name"])

    # Only try alt_names if name search failed
    alt_names = map_alt_names_to_list(manwhas_df)
    most_similar_alt_name, highest_alt_similarity = get_close_matches(
        target_manhwa, alt_names, limit=1
    )[0]

    # Only proceed if alt name search is better than name search
    if highest_alt_similarity > highest_similarity and highest_alt_similarity > 0.70:
        mask = manwhas_df["altName"].str.contains(
            f"\\b{most_similar_alt_name}\\b", case=False, na=False, regex=True
        )
        target_manwha_df = manwhas_df[mask]
        return (target_manwha_df.index[0], target_manwha_df["name"])

    raise IndexError(
        f'Exact match for {target_manhwa} not found. The closest match is "{most_similar_name}" with a similarity ratio of {highest_similarity}'
    )
```

**Better**: Consider caching expanded alt_names or using approximate nearest neighbor search for large datasets.

---

### 7. Duplicate Code - DRY Violation (File I/O)
**Files**: Multiple
**Severity**: MEDIUM-HIGH

**Pattern repeated in 3+ places**:

1. `scripts/get_cleaned_manwha_recommendations.py:15-16`:
```python
with open("data/raw_manwha_reddit_recommendations.json", "r") as f:
    return json.load(f)
```

2. `scripts/parse_raw_reddit_recommendations.py:229-231`:
```python
with open(RAW_REDDIT_RECOMMENDATIONS_TXT_FILE_PATH, "r") as f:
    lines = f.readlines()
    return [line.strip() for line in lines if line.strip() != ""]
```

3. `src/utils/manwha_utils.py:119-121`:
```python
with open(CLEANED_MANWHAS_PATH, "r") as f:
    data = json.load(f)
    return preprocess_manwhas(data)
```

**Fix**: Create `src/utils/file_io.py`:
```python
import json
from pathlib import Path
from typing import Any, List

def load_json(path: Path) -> dict:
    """Load JSON file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

def save_json(path: Path, data: Any, indent: int = 4) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)

def load_text_lines(path: Path) -> List[str]:
    """Load text file as list of non-empty lines."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Text file not found: {path}")
```

Then replace all file I/O code with these utilities.

---

## 🟢 MEDIUM PRIORITY ISSUES

### 8. Missing Data Validation
**Files**: All parser files
**Severity**: MEDIUM

**Issue**: No validation that loaded JSON has expected structure. What happens if:
- `data/rawManwhas.json` is corrupted?
- Required fields are missing?
- Data types are wrong?
- Array fields contain non-array values?

**Fix**: Add validation with Pydantic or dataclasses:
```python
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
        """Validate data after initialization."""
        if not self.name:
            raise ValueError("Manwha name cannot be empty")
        if not 0 <= self.rating <= 5:
            raise ValueError(f"Rating must be 0-5, got {self.rating}")
        if not isinstance(self.tags, list):
            raise ValueError("Tags must be a list")

    @classmethod
    def from_dict(cls, data: dict) -> 'ManwhaData':
        """Create from dictionary with validation."""
        try:
            return cls(
                name=data["name"],
                rating=float(data.get("rating", 0)),
                tags=data.get("tags", []),
                description=data.get("description", ""),
                altName=data.get("altName"),
                chapters=data.get("chapters"),
            )
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
```

---

### 9. Shadowing Built-in Name
**File**: `scripts/parse_raw_reddit_recommendations.py`
**Line**: 94
**Severity**: MEDIUM

```python
def is_alt_name(str: str):  # ❌ 'str' shadows built-in!
    return str.lower().startswith("or,")
```

**Issue**: Parameter named `str` shadows Python's built-in `str` type. This can cause confusing bugs.

**Fix**:
```python
def is_alt_name(text: str) -> bool:
    """Check if the string is an alt name (starts with 'or,')."""
    return text.lower().startswith("or,")
```

---

### 10. Unused Imports
**File**: `scripts/parse_raw_reddit_recommendations.py`
**Line**: 62
**Severity**: LOW

```python
import pandas as pd  # ❌ Never used in this file
```

Also line 1:
```python
from typing import List, Dict, Set  # 'Set' is never used
```

**Fix**: Remove unused imports:
```python
from typing import List, Dict
```

---

### 11. Magic Numbers Without Named Constants
**File**: `scripts/parse_raw_reddit_recommendations.py`
**Lines**: 32-38
**Severity**: MEDIUM

```python
RATINGS = {
    "P": 5.0,  # What do these numbers mean?
    "E": 4.0,
    "G": 3.0,
    "S": 2.0,
    "D": 1.0,
}
```

**Fix**: Use descriptive names:
```python
RATING_PEAK = 5.0
RATING_ENJOYED = 4.0
RATING_GOOD = 3.0
RATING_SUCKS = 2.0
RATING_DROPPED = 1.0

RATINGS = {
    "P": RATING_PEAK,
    "E": RATING_ENJOYED,
    "G": RATING_GOOD,
    "S": RATING_SUCKS,
    "D": RATING_DROPPED,
}
```

**File**: `src/utils/manwha_utils.py`
**Lines**: 70, 82
**Severity**: MEDIUM

```python
if highest_similarity > 0.70:  # Magic number
```

**Fix**:
```python
SIMILARITY_THRESHOLD = 0.70  # Minimum similarity for fuzzy match

if highest_similarity > SIMILARITY_THRESHOLD:
```

---

### 12. Missing Docstrings
**Files**: Multiple
**Severity**: MEDIUM

**Missing or inadequate docstrings**:
- `scripts/build.py:run()` - What does this do? What order? What files created?
- `scripts/get_cleaned_manwha_recommendations.py:map_reddit_recommendations()` - Complex logic needs explanation
- `src/utils/manwha_utils.py:preprocess_manwhas()` - What preprocessing happens?
- `scripts/parse_raw_reddit_recommendations.py:preprocess()` - What data transformations occur?

**Example Fix** for `scripts/build.py`:
```python
def run():
    """Build the manwha recommender system.

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
    """
```

---

## 🔵 LOW PRIORITY / STYLE ISSUES

### 13. Inconsistent Return Types
**File**: `src/utils/manwha_utils.py`
**Lines**: 98-103
**Severity**: LOW

```python
def find_manwha_by_name(manwhas_df: pd.DataFrame, name: str):
    try:
        # Returns pd.Series
        return manwhas_df.iloc[target_index]
    except IndexError as e:
        # Returns str
        return MANWHA_NOT_FOUND
```

**Issue**: Function returns EITHER `pd.Series` OR `str`. This makes type hints useless and forces runtime type checking.

**Fix**: Already covered in issue #2 - use Optional[pd.Series].

---

### 14. Commented Code Doesn't Match Behavior
**File**: `scripts/parse_manwha.py`
**Line**: 38-40
**Severity**: LOW

```python
# remove non-printable characters using char for char in text if char in cleaned_text
cleaned_text = [char for char in text if char in text]
return "".join(cleaned_text)
```

**Issue**: Comment claims to filter non-printable characters, but code doesn't do that (covered in issue #1).

---

### 15. Inconsistent Code Style
**Files**: Multiple
**Severity**: LOW

**Issues**:
1. Mixed string quotes (`"` vs `'`)
2. Inconsistent f-string usage
3. Inconsistent spacing around operators

**Recommendation**: Run `black` formatter and `flake8` linter:
```bash
pip install black flake8
black .
flake8 . --max-line-length=100
```

Add to CI/CD pipeline to enforce consistent style.

---

## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| Total Issues | 15 |
| Critical | 3 |
| High Priority | 4 |
| Medium Priority | 6 |
| Low Priority | 2 |
| Lines Changed | ~29,426 |
| Test Coverage | ~5% |
| Required Coverage | 70%+ |

---

## ✅ Positive Aspects

1. Good refactoring: Extracting utilities from recommender improves separation of concerns
2. Type hints present on most functions
3. Meaningful variable and function names
4. Clear code intent despite issues
5. Useful functionality (Reddit recommendations integration)

---

## 🎯 Required Actions Before Merge

### MUST FIX (Blocking):
1. ✅ Fix critical bug in `clean_text()` (issue #1)
2. ✅ Replace string sentinel values with Optional types (issue #2)
3. ✅ Add proper error handling with logging (issue #3)
4. ✅ Add comprehensive tests - minimum 70% coverage (issue #4)
5. ✅ Centralize all path constants (issue #5)

### SHOULD FIX (Strongly Recommended):
6. ✅ Fix performance issue in `get_target_manwha()` (issue #6)
7. ✅ Create file I/O utility module (issue #7)
8. ✅ Add data validation (issue #8)
9. ✅ Fix shadowed built-in name (issue #9)
10. ✅ Remove unused imports (issue #10)

### NICE TO HAVE:
11. ⚠️ Use named constants for magic numbers (issue #11)
12. ⚠️ Add comprehensive docstrings (issue #12)
13. ⚠️ Run black formatter for consistent style (issue #15)

---

**Overall Recommendation**: ❌ **REQUEST CHANGES** - Critical bugs and lack of tests are blocking issues.
