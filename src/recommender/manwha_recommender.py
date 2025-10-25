import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
from src.utils.constants import MANWHA_NOT_FOUND, MANWHA_RECOMMENDER_MODEL_PATH
from src.utils.manwha_utils import get_target_manwha, get_manwhas


class _ManwhaRecommender:
    def __init__(self):
        (
            self._knn_model,
            self._feature_matrix,
            self._manwhas_df,
        ) = self._get_manwhas_knn_model()

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
        manwhas_df = get_manwhas()
        tfidf_matrix = self._vectorize_manwhas(manwhas_df)
        scaled_manwhas_df = self._scale_ratings(manwhas_df)
        combined_features_matrix = self._combine_features(tfidf_matrix, scaled_manwhas_df)
        model = self._build_knn_model(combined_features_matrix)
        return model, combined_features_matrix, scaled_manwhas_df

    def recommend(self, input_manhwa_name):
        try:
            target_index, target_manwha = get_target_manwha(self._manwhas_df, input_manhwa_name)
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
    with open(MANWHA_RECOMMENDER_MODEL_PATH, "wb") as file:
        pickle.dump(_ManwhaRecommender(), file)
        print("Manwha recommender model saved.")


def load_manwha_recommender_model() -> _ManwhaRecommender:
    with open(MANWHA_RECOMMENDER_MODEL_PATH, "rb") as file:
        return pickle.load(file)


__all__ = [
    "save_manwha_recommender_model",
    "load_manwha_recommender_model",
]
