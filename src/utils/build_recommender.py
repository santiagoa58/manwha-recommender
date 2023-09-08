from src.recommender.manwha_recommender import save_manwha_recommender_model
import src.utils.parse_manwha as parse_manwha


def run():
    parse_manwha.run()
    save_manwha_recommender_model()


if __name__ == "__main__":
    run()
