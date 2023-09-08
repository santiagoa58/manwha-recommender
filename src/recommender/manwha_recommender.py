import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
import json
import difflib

MANWHA_NOT_FOUND = "The manhwa you entered is not in our database."
MODEL_PATH = "models/manwha_recommender.pkl"


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

    def get_target_manwha(self, target_manhwa):
        # Preprocess the target name
        processed_target = target_manhwa.lower().replace(" ", "")

        # Calculate similarity for each name in the DataFrame
        similarities = [
            (
                name,
                difflib.SequenceMatcher(
                    None, processed_target, name.lower().replace(" ", "")
                ).ratio(),
            )
            for name in self._manwhas_df["name"].tolist()
        ]

        # Sort by similarity and pick the most similar name
        most_similar_name, highest_similarity = sorted(
            similarities, key=lambda x: x[1], reverse=True
        )[0]

        # You can adjust the threshold (0.7 in this example) based on your requirements.
        if highest_similarity > 0.7:
            return (
                self._manwhas_df[self._manwhas_df["name"] == most_similar_name].index[
                    0
                ],
                most_similar_name,
            )

        raise IndexError("No close match found.")

    def recommend(self, target_manhwa):
        try:
            target_index, most_similar_target_name = self.get_target_manwha(
                target_manhwa
            )
            _, indices = self._knn_model.kneighbors(
                self._feature_matrix.tocsr()[target_index], n_neighbors=11
            )
            similar_manwhas_indices = indices.flatten()[1:]
            similar_manwhas_names = self._manwhas_df.iloc[similar_manwhas_indices][
                "name"
            ]
            return similar_manwhas_names.tolist(), most_similar_target_name
        except IndexError:
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
