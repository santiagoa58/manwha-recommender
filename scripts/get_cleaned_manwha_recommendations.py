from pandas import DataFrame
from src.utils.constants import (
    MANWHA_NOT_FOUND,
    RAW_REDDIT_RECOMMENDATIONS_JSON,
    CLEANED_REDDIT_RECOMMENDATIONS,
    UNCLEANED_REDDIT_RECOMMENDATIONS,
)
from src.utils.manwha_utils import find_manwha_by_name, get_manwhas
from src.utils.file_io import load_json, save_json


def load_manwha_reddit_recommendations() -> dict:
    return load_json(RAW_REDDIT_RECOMMENDATIONS_JSON)


def map_reddit_recommendations(all_manwhas: DataFrame, reddit_manwhas: dict):
    """Map Reddit recommendations to manwha database entries.

    Loads raw Reddit recommendations and cross-references them with
    the cleaned manwha database. Recommendations that match are added
    to the cleaned list; unmatched ones go to the uncleaned list.

    Args:
        all_manwhas: DataFrame containing all manwha database entries
        reddit_manwhas: Dictionary of Reddit recommendations keyed by title

    Returns:
        Tuple containing:
        - recommendations: List of recommendations matched to database
        - not_found: List of recommendations not found in database

    Raises:
        IndexError: If manwha lookup fails unexpectedly
    """
    recommendations = []
    not_found = []
    for manwha_name, manwha in reddit_manwhas.items():
        manwha_details = find_manwha_by_name(all_manwhas, manwha_name)
        if manwha_details is None:
            manwha["title"] = manwha_name
            not_found.append(manwha)
            continue

        manwha_cp = manwha_details.to_dict()
        manwha_cp["recommendation_rating"] = manwha["rating"]
        manwha_cp["recommendation_notes"] = manwha["notes"]
        manwha_cp["recommendation_categories"] = manwha["categories"]
        manwha_cp["recommendation_subcategories"] = manwha["subcategories"]
        manwha_cp["recommendation_alt_names"] = manwha["alt_names"]
        recommendations.append(manwha_cp)
    return recommendations, not_found


def run():
    print("loading parsed reddit recommendations...")
    reddit_recommendations = load_manwha_reddit_recommendations()
    all_manwhas_df = get_manwhas()
    print("finished loading!")
    print("cross referencing parsed reddit recommendations with existing manwha data...")
    recommendations, manwhas_not_cleaned = map_reddit_recommendations(
        all_manwhas_df, reddit_recommendations
    )
    print("Done!")
    save_json(CLEANED_REDDIT_RECOMMENDATIONS, recommendations)
    print(
        f"{len(recommendations)} manwhas were mapped correctly and written to: {CLEANED_REDDIT_RECOMMENDATIONS}"
    )

    save_json(UNCLEANED_REDDIT_RECOMMENDATIONS, manwhas_not_cleaned)
    print(
        f"{len(manwhas_not_cleaned)} manwhas could not be mapped. results were written to: {UNCLEANED_REDDIT_RECOMMENDATIONS}"
    )


if __name__ == "__main__":
    run()
