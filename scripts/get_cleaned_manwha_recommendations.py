from pandas import DataFrame
import json
from src.utils.constants import MANWHA_NOT_FOUND
from src.utils.manwha_utils import find_manwha_by_name, get_manwhas

CLEANED_REDDIT_MANWHA_RECOMMENDATIONS_PATH = (
    "data/cleaned_manwha_reddit_recommendations.json"
)
UNCLEANED_REDDIT_MANWHA_RECOMMENDATIONS_PATH = (
    "data/uncleaned_manwha_reddit_recommendations.json"
)


def load_manwha_reddit_recommendations() -> dict:
    with open("data/raw_manwha_reddit_recommendations.json", "r") as f:
        return json.load(f)


def map_reddit_recommendations(all_manwhas: DataFrame, reddit_manwhas: dict):
    """Map the reddit recommendations to the manwhas"""
    recommendations = []
    not_found = []
    for manwha_name, manwha in reddit_manwhas.items():
        try:
            manwha_details = find_manwha_by_name(all_manwhas, manwha_name)
            if type(manwha_details) is str and manwha_details == MANWHA_NOT_FOUND:
                manwha["title"] = manwha_name
                not_found.append(manwha)
                continue
            else:
                manwha_cp = manwha_details.to_dict()
                manwha_cp["recommendation_rating"] = manwha["rating"]
                manwha_cp["recommendation_notes"] = manwha["notes"]
                manwha_cp["recommendation_categories"] = manwha["categories"]
                manwha_cp["recommendation_subcategories"] = manwha["subcategories"]
                manwha_cp["recommendation_alt_names"] = manwha["alt_names"]
                recommendations.append(manwha_cp)
        except IndexError:
            not_found.append(manwha)
            continue
    return recommendations, not_found


def run():
    print("loading parsed reddit recommendations...")
    reddit_recommendations = load_manwha_reddit_recommendations()
    all_manwhas_df = get_manwhas()
    print("finished loading!")
    print(
        "cross referencing parsed reddit recommendations with existing manwha data..."
    )
    recommendations, manwhas_not_cleaned = map_reddit_recommendations(
        all_manwhas_df, reddit_recommendations
    )
    print("Done!")
    with open(CLEANED_REDDIT_MANWHA_RECOMMENDATIONS_PATH, "w") as f:
        json.dump(recommendations, f, indent=4)
        print(
            f"{len(recommendations)} manwhas were mapped correctly and written to: {CLEANED_REDDIT_MANWHA_RECOMMENDATIONS_PATH}"
        )

    with open(UNCLEANED_REDDIT_MANWHA_RECOMMENDATIONS_PATH, "w") as f:
        json.dump(manwhas_not_cleaned, f, indent=4)
        print(
            f"{len(manwhas_not_cleaned)} manwhas could not be mapped. results were written to: {UNCLEANED_REDDIT_MANWHA_RECOMMENDATIONS_PATH}"
        )


if __name__ == "__main__":
    run()
