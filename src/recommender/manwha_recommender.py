import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
import json
import difflib
from src.utils.constants import MANWHA_NOT_FOUND

MODEL_PATH = "models/manwha_recommender.pkl"


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


class _ManwhaRecommender:
    def __init__(self):
        (
            self._knn_model,
            self._feature_matrix,
            self._manwhas_df,
        ) = self._get_manwhas_knn_model()

    def _preprocess_text(self, text):
        return str(text).lower().replace("\n", " ").replace("\r", "")

    def _preprocess_manwhas(self, manwhas_json):
        cleaned_manwhas_df = pd.DataFrame(manwhas_json)
        cleaned_manwhas_df["description"] = cleaned_manwhas_df["description"].apply(
            self._preprocess_text
        )
        cleaned_manwhas_df["tags"] = cleaned_manwhas_df["tags"].apply(
            lambda x: " ".join(x)
        )
        cleaned_manwhas_df["combined_text"] = (
            cleaned_manwhas_df["description"] + " " + cleaned_manwhas_df["tags"]
        )
        return cleaned_manwhas_df

    def _get_manwhas(self):
        with open("./data/cleanedManwhas.json", "r") as f:
            data = json.load(f)
            return self._preprocess_manwhas(data)

    def _vectorize_manwhas(self, manwhas_df):
        tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        return tfidf_vectorizer.fit_transform(manwhas_df["combined_text"])

    def _scale_ratings(self, manwhas_df):
        scaler = MinMaxScaler()
        manwhas_df["scaled_rating"] = scaler.fit_transform(manwhas_df[["rating"]])
        return manwhas_df

    def _combine_features(self, tfidf_matrix, manwhas_df):
        return hstack([tfidf_matrix, manwhas_df["scaled_rating"].values.reshape(-1, 1)])

    def _build_knn_model(self, features_matrix):
        return NearestNeighbors(n_neighbors=11, algorithm="auto").fit(features_matrix)

    def _get_manwhas_knn_model(self):
        manwhas_df = self._get_manwhas()
        tfidf_matrix = self._vectorize_manwhas(manwhas_df)
        scaled_manwhas_df = self._scale_ratings(manwhas_df)
        combined_features_matrix = self._combine_features(
            tfidf_matrix, scaled_manwhas_df
        )
        model = self._build_knn_model(combined_features_matrix)
        return model, combined_features_matrix, scaled_manwhas_df

    def get_target_manwha(self, target_manhwa) -> tuple[int, str]:
        most_similar_name, highest_similarity = get_close_matches(
            target_manhwa, self._manwhas_df["name"].tolist(), limit=1
        )[0]

        if highest_similarity > 0.70:
            target_manwha_df = self._manwhas_df[
                self._manwhas_df["name"] == most_similar_name
            ]
            return (
                target_manwha_df.index[0],
                most_similar_name,
            )

        raise IndexError(
            f'Exact match for {target_manhwa} not found. The closest match is "{most_similar_name}" with a similarity ratio of {highest_similarity}'
        )

    def recommend(self, input_manhwa_name):
        try:
            target_index, target_manwha = self.get_target_manwha(input_manhwa_name)
            _, indices = self._knn_model.kneighbors(
                self._feature_matrix.tocsr()[target_index], n_neighbors=11
            )
            similar_manwhas_indices = indices.flatten()[1:]
            similar_manwhas = self._manwhas_df.iloc[similar_manwhas_indices]
            # return only the name, alt_name, rating, tags, description, source, publisher, years, chapters, volumes, url and filter out the generated columns
            similar_manwhas = similar_manwhas[
                [
                    "name",
                    "altName",
                    "rating",
                    "tags",
                    "description",
                    "source",
                    "publisher",
                    "years",
                    "chapters",
                    "imageURL",
                    "id",
                ]
            ]
            return similar_manwhas, target_manwha
        except IndexError as e:
            print(e)
            return MANWHA_NOT_FOUND


def save_manwha_recommender_model():
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(_ManwhaRecommender(), file)
        print("Manwha recommender model saved.")


def load_manwha_recommender_model() -> _ManwhaRecommender:
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


__all__ = [
    "save_manwha_recommender_model",
    "load_manwha_recommender_model",
    "MANWHA_NOT_FOUND",
]
