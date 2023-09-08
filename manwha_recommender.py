import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
import json


# Function to preprocess text
def preprocess_text(text):
    return str(text).lower().replace("\n", " ").replace("\r", "")


# Function to preprocess manwhas data
def preprocess_manwhas(manwhas_json):
    cleaned_manwhas_df = pd.DataFrame(manwhas_json)
    cleaned_manwhas_df["description"] = cleaned_manwhas_df["description"].apply(
        preprocess_text
    )
    cleaned_manwhas_df["tags"] = cleaned_manwhas_df["tags"].apply(lambda x: " ".join(x))
    # Combine 'description' and 'tags' into a single feature
    cleaned_manwhas_df["combined_text"] = (
        cleaned_manwhas_df["description"] + " " + cleaned_manwhas_df["tags"]
    )
    return cleaned_manwhas_df


# Read in cleaned manwhas data and preprocess it
def get_manwhas():
    with open("cleanedManwhas.json", "r") as f:
        data = json.load(f)
        return preprocess_manwhas(data)


# TF-IDF Vectorizer for text features
def vectorize_manwhas(manwhas_df):
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    return tfidf_vectorizer.fit_transform(manwhas_df["combined_text"])


# MinMax Scaling for 'rating'
def scale_ratings(manwhas_df):
    scaler = MinMaxScaler()
    manwhas_df["scaled_rating"] = scaler.fit_transform(manwhas_df[["rating"]])
    return manwhas_df


# Combine text features and scaled 'rating' feature
def combine_features(tfidf_matrix, manwhas_df):
    return hstack([tfidf_matrix, manwhas_df["scaled_rating"].values.reshape(-1, 1)])


def build_knn_model(features_matrix):
    return NearestNeighbors(n_neighbors=11, algorithm="auto").fit(features_matrix)


def get_manwhas_knn_model():
    manwhas_df = get_manwhas()
    tfidf_matrix = vectorize_manwhas(manwhas_df)
    scaled_manwhas_df = scale_ratings(manwhas_df)
    combined_features_matrix = combine_features(tfidf_matrix, scaled_manwhas_df)
    model = build_knn_model(combined_features_matrix)
    return model, combined_features_matrix, scaled_manwhas_df


def get_target_manwha_index(target_manhwa, manwhas_df):
    return manwhas_df[manwhas_df["name"].str.lower() == target_manhwa.lower()].index[0]


def recommend_manhwa(
    target_manhwa,
    knn_model,
    feature_matrix,
    df,
):
    try:
        # Find the index of the target manhwa
        target_index = get_target_manwha_index(target_manhwa, df)

        # Use KNN model to find similar manhwas
        distances, indices = knn_model.kneighbors(
            feature_matrix.tocsr()[target_index], n_neighbors=11
        )

        # Extract indices of similar manhwas and their names
        similar_manwhas_indices = indices.flatten()[1:]
        similar_manwhas_names = df.iloc[similar_manwhas_indices]["name"]

        # Return the list of recommended manhwas
        return similar_manwhas_names.tolist()

    except IndexError:
        return "The manhwa you entered is not in our database."


# Build the recommender
def build_manwha_recommender():
    knn_model, feature_matrix, df = get_manwhas_knn_model()
    return lambda target_manhwa: recommend_manhwa(
        target_manhwa, knn_model, feature_matrix, df
    )


manwha_recommender_system = build_manwha_recommender()

### Making recommendations for a given manhwa ###


# command line interactions, let user provide name of a manwha and get recommendations
def main():
    parser = argparse.ArgumentParser(
        description="Recommend manhwas similar to the provided manhwa."
    )
    parser.add_argument(
        "manhwa_name",
        type=str,
        help="The full name of the manhwa for which to get recommendations.",
    )

    args = parser.parse_args()
    recommendations = manwha_recommender_system(args.manhwa_name)

    print(f"Recommendations for {args.manhwa_name}:")
    for rec in recommendations:
        print(rec)


if __name__ == "__main__":
    main()
