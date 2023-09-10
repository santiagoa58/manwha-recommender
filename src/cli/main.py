from pandas import DataFrame
from src.recommender.manwha_recommender import (
    load_manwha_recommender_model,
)
from src.utils.constants import MANWHA_NOT_FOUND, UNKNOWN

RECOMENDER = load_manwha_recommender_model()


def print_manwha_info(manwha_name_df: DataFrame):
    # sort by rating
    manwha_name_df = manwha_name_df.sort_values(by="rating", ascending=False)
    for _, row in manwha_name_df.iterrows():
        alt_name = row["altName"]
        name = (
            f'{row["name"]} (AKA: {alt_name})'
            if alt_name and alt_name != UNKNOWN
            else row["name"]
        )
        chapters_and_volumes: dict = row["chapters"]
        volumes = chapters_and_volumes["volumes"]
        chapters = chapters_and_volumes["chapters"]
        chapters = (
            f"Chapters: {chapters_and_volumes['chapters']}, Volumes: {volumes}"
            if volumes and volumes != UNKNOWN
            else f"Chapters: {chapters}"
        )
        print(f"{name}")
        print(f"\tRating: {row['rating']}")
        print(f"\tTags: {row['tags']}")
        print(f"\tYears Active: {row['years']}")
        print(f"\t{chapters}")
        print("-" * 50)  # Just a separator for readability


def main():
    while True:
        # Get manwha name from the user
        manhwa_name = input(
            "Enter the name of the manwha (or type 'exit' or 'q' or 'quit' to quit): "
        )

        # Check if the user wants to exit
        if manhwa_name.lower() in ["exit", "q", "quit"]:
            break

        # Get recommendations
        results = RECOMENDER.recommend(manhwa_name)
        # Display recommendations
        if results == MANWHA_NOT_FOUND:
            print(results)
        else:
            recommendations, target_manwha_name = results
            print(f"\nRecommended Manwhas for {target_manwha_name}:\n")
            print_manwha_info(recommendations)
        print("\n")


if __name__ == "__main__":
    main()
