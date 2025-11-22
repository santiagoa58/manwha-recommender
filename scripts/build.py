import logging
from src.recommender.manwha_recommender import save_manwha_recommender_model
import scripts.parse_manwha as parse_manwha
import scripts.parse_raw_reddit_recommendations as parse_raw_recommendations
import scripts.get_cleaned_manwha_recommendations as clean_manwha_recommendations

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    run()
