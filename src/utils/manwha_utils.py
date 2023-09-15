import pandas as pd
import json
import difflib
from src.utils.constants import MANWHA_NOT_FOUND, UNKNOWN, CLEANED_MANWHAS_PATH


def split_alt_names(manwha_df: pd.DataFrame):
    return manwha_df["altName"].str.split(",", expand=True)


def clean_alt_name(alt_name: str):
    return alt_name.strip() if alt_name else UNKNOWN


def map_alt_names_to_list(manwha_df: pd.DataFrame) -> list[str]:
    alt_names = split_alt_names(manwha_df).stack().tolist()
    return [
        clean_alt_name(alt_name)
        for alt_name in alt_names
        if clean_alt_name(alt_name) != UNKNOWN
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


def get_target_manwha(manwhas_df: pd.DataFrame, target_manhwa: str) -> tuple[int, str]:
    most_similar_name, highest_similarity = get_close_matches(
        target_manhwa, manwhas_df["name"].tolist(), limit=1
    )[0]

    if highest_similarity > 0.70:
        target_manwha_df = manwhas_df[manwhas_df["name"] == most_similar_name]
        return (
            target_manwha_df.index[0],
            target_manwha_df["name"],
        )
    # try searching for alt_names
    alt_names = map_alt_names_to_list(manwhas_df)
    most_similar_alt_name, highest_alt_similarity = get_close_matches(
        target_manhwa, alt_names, limit=1
    )[0]

    if highest_similarity < highest_alt_similarity and highest_alt_similarity > 0.70:
        # Splits altnames like "altname1, altname2" into ["altname1", "altname2"] and checks for a match.
        mask = manwhas_df["altName"].str.contains(
            f"\\b{most_similar_alt_name}\\b", case=False, na=False, regex=True
        )
        target_manwha_df = manwhas_df[mask]
        return (
            target_manwha_df.index[0],
            target_manwha_df["name"],
        )

    raise IndexError(
        f'Exact match for {target_manhwa} not found. The closest match is "{most_similar_name}" with a similarity ratio of {highest_similarity}'
    )


def find_manwha_by_name(manwhas_df: pd.DataFrame, name: str):
    try:
        target_index, target_name = get_target_manwha(manwhas_df, name)
        return manwhas_df.iloc[target_index]
    except IndexError as e:
        return MANWHA_NOT_FOUND


def preprocess_manwhas(manwhas_json):
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
    with open(CLEANED_MANWHAS_PATH, "r") as f:
        data = json.load(f)
        return preprocess_manwhas(data)
