from src.recommender.manwha_recommender import save_manwha_recommender_model
import scripts.parse_manwha as parse_manwha
import scripts.parse_raw_reddit_recommendations as parse_raw_recommendations
import scripts.get_cleaned_manwha_recommendations as clean_manwha_recommendations


def run():
    parse_manwha.run()
    parse_raw_recommendations.run()
    clean_manwha_recommendations.run()
    save_manwha_recommender_model()


if __name__ == "__main__":
    run()
