from src.recommender.manwha_recommender import (
    load_manwha_recommender_model,
    MANWHA_NOT_FOUND,
)


RECOMENDER = load_manwha_recommender_model()


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
            print(f"\nRecommended Manwhas for {target_manwha_name}:")
            for idx, rec in enumerate(recommendations, 1):
                print(f"{idx}. {rec}")
        print("\n")


if __name__ == "__main__":
    main()
