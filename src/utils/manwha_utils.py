import pandas as pd
import difflib
from typing import Optional
from src.utils.constants import MANWHA_NOT_FOUND, UNKNOWN, CLEANED_MANWHAS_PATH
from src.utils.file_io import load_json


def split_alt_names(manwha_df: pd.DataFrame):
    return manwha_df["altName"].str.split(",", expand=True)


def clean_alt_name(alt_name: str):
    return alt_name.strip() if alt_name else UNKNOWN


def map_alt_names_to_list(manwha_df: pd.DataFrame) -> list[str]:
    alt_names = split_alt_names(manwha_df).stack().tolist()
    return [
        clean_alt_name(alt_name) for alt_name in alt_names if clean_alt_name(alt_name) != UNKNOWN
    ]


def get_similarity_ratio(target: str, source: str):
    """
    Returns the similarity ratio between two strings.
    """
    return difflib.SequenceMatcher(
        None, target.lower().replace(" ", ""), source.lower().replace(" ", "")
    ).ratio()


def get_close_matches(target: str, sources: list[str], limit=1):
    """
    Returns a list of tuples containing the name and similarity ratio of the most similar names.

    Example:
        >>> get_close_matches("solo leveln", ["solo leveling", "tower of god", "the god of high school", "solo leveling 3", "nano machine", "solo leveling 2",], limit=1)
        [('solo leveling', 0.93333)]

    Args:
        target (str): The target name.
        sources (list[str]): A list of names to compare the target name to.
        limit (int, optional): The number of similar names to return. Defaults to 1.

    Returns:
        list[tuple[str, float]]: A list of tuples containing the name and similarity ratio of the most similar names.
    """
    similarities = [
        (
            name,
            get_similarity_ratio(target, name),
        )
        for name in sources
    ]

    # Sort by similarity and pick the most similar name
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return [(name, similarity) for name, similarity in sorted_similarities[:limit]]


import pandas as pd

# Minimum similarity threshold for fuzzy matching (70%)
SIMILARITY_THRESHOLD = 0.70


def get_target_manwha(manwhas_df: pd.DataFrame, target_manhwa: str) -> tuple[int, str]:
    """Find manwha by name using fuzzy matching.

    Searches names first, then alt names only if needed.

    Args:
        manwhas_df: DataFrame with manwha data
        target_manhwa: Name to search for

    Returns:
        Tuple of (index, name)

    Raises:
        IndexError: If no match found above threshold
    """
    # First try exact name matches
    most_similar_name, highest_similarity = get_close_matches(
        target_manhwa, manwhas_df["name"].tolist(), limit=1
    )[0]

    if highest_similarity > SIMILARITY_THRESHOLD:
        target_manwha_df = manwhas_df[manwhas_df["name"] == most_similar_name]
        return (
            target_manwha_df.index[0],
            target_manwha_df["name"].iloc[0],
        )

    # Only search alt_names if name search failed
    alt_names = map_alt_names_to_list(manwhas_df)
    most_similar_alt_name, highest_alt_similarity = get_close_matches(
        target_manhwa, alt_names, limit=1
    )[0]

    # Only proceed if alt name search is better AND above threshold
    if highest_alt_similarity > SIMILARITY_THRESHOLD:
        # Splits altnames like "altname1, altname2" into ["altname1", "altname2"] and checks for a match.
        mask = manwhas_df["altName"].str.contains(
            f"\\b{most_similar_alt_name}\\b", case=False, na=False, regex=True
        )
        target_manwha_df = manwhas_df[mask]
        return (
            target_manwha_df.index[0],
            target_manwha_df["name"].iloc[0],
        )

    # Neither search succeeded
    raise IndexError(
        f'Exact match for "{target_manhwa}" not found. '
        f'Closest match is "{most_similar_name}" with similarity {highest_similarity:.2f}'
    )


def find_manwha_by_name(manwhas_df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Find a manwha by name or alt name.

    Args:
        manwhas_df: DataFrame containing manwha data
        name: Manwha name to search for

    Returns:
        pd.Series if found, None if not found
    """
    try:
        target_index, target_name = get_target_manwha(manwhas_df, name)
        return manwhas_df.iloc[target_index]
    except IndexError:
        return None


def preprocess_manwhas(manwhas_json):
    """Preprocess raw manwha data into DataFrame.

    Converts list of manwha dictionaries into a pandas DataFrame
    with proper data types and cleaned columns. Combines description
    and tags into a single 'combined_text' field for ML processing.

    Args:
        manwhas_json: List of manwha dictionaries from JSON

    Returns:
        DataFrame with columns: name, rating, tags, description, altName,
        combined_text, etc.

    Raises:
        ValueError: If data is empty or malformed
    """
    cleaned_manwhas_df = pd.DataFrame(manwhas_json)
    cleaned_manwhas_df["description"] = cleaned_manwhas_df["description"].apply(
        lambda text: str(text).lower().replace("\n", " ").replace("\r", "")
    )
    cleaned_manwhas_df["tags"] = cleaned_manwhas_df["tags"].apply(lambda x: " ".join(x))
    cleaned_manwhas_df["combined_text"] = (
        cleaned_manwhas_df["description"] + " " + cleaned_manwhas_df["tags"]
    )
    return cleaned_manwhas_df


def get_manwhas():
    data = load_json(CLEANED_MANWHAS_PATH)
    return preprocess_manwhas(data)
