from typing import List, Dict, Set
import json
import re

NO_SUBCATEGORY = None
# Dictionary of terms to know
DICTIONARY = {
    "manhwa": "Korean comic",
    "manhua": "Chinese comic",
    "manga": "Japanese comic",
}

RATINGS_DICTIONARY = {
    "P": ["Best", "Peak", "Best/Peak Manhwa"],
    "E": [
        "Decent/Enjoyed Reading It/Entertaining",
        "Enjoyed",
        "Decent",
        "Enjoyed Reading It",
        "Entertaining",
    ],
    "G": ["Good Enough To Pass Time", "Good", "Good Enough"],
    "S": [
        "Sucks",
        "Kinda sucks (but only halfway)",
        "Kinda Sucks",
        "Sucks (but only halfway)",
    ],
    "D": ["Not good/Dropped", "Dropped", "Not good"],
}

RATINGS = {
    "P": 5.0,
    "E": 4.0,
    "G": 3.0,
    "S": 2.0,
    "D": 1.0,
}

RAW_REDDIT_RECOMMENDATIONS_TXT_FILE_PATH = "data/raw_manwha_reddit_recommendations.txt"
RAW_REDDIT_RECOMMENDATIONS_JSON_FILE_PATH = (
    "data/raw_manwha_reddit_recommendations.json"
)


def get_rating_ratio(rating_ratio: str) -> float:
    if "/" in rating_ratio:
        ratings = [rating for rating in rating_ratio.split("/") if rating]
        numerical_ratings = []
        for rating in ratings:
            numerical_rating = get_rating(rating)
            numerical_ratings.append(numerical_rating) if numerical_rating else None
        if len(numerical_ratings) == 0:
            return None
        return sum(numerical_ratings) / len(numerical_ratings)


def get_rating(rating: str) -> float:
    """Get the numerical rating if the provided string is a valid rating"""
    rating = rating.strip().upper()
    if rating in RATINGS:
        return RATINGS[rating]

    # go through RATINGS_DICTIONARY and check if the rating is in the list of ratings
    for key, values in RATINGS_DICTIONARY.items():
        if rating in [val.upper() for val in values]:
            return RATINGS[key]
    # if rating is a ratio, return the numerical rating as an average of the two ratings
    if "/" in rating:
        return get_rating_ratio(rating)

    return None


def get_rating_from_title(title: str):
    """Get the rating from the title"""
    # example titles: Solo leveling (P),  return of the mount hua sect (comedy) (E), legend of the northern blade (peak)
    # extract the rating from the title, the last (...) in the title is the rating
    possible_rating = title.split("(")[-1].strip(" )")
    return get_rating(possible_rating)


def get_notes_from_title(title: str):
    """Gets the notes from the title"""
    # example titles: Solo leveling (P),  return of the mount hua sect (comedy) (E), legend of the northern blade (peak)
    # extract the notes from the title, the last (...) in the title are the notes, the ratings are omitted
    possible_notes = re.findall(r"\((.*?)\)", title)
    notes = [note for note in possible_notes if get_rating(note) is None]
    if len(notes) != 0:
        return notes
    return None


def is_alt_name(str: str):
    """Check if the string is an alt name"""
    # alt names are surrounded by ""
    return str.lower().startswith("or,")


def get_alt_names_from_notes(notes: List[str]):
    """Gets the alt name from the notes"""
    # alt name starts with or,
    alt_names = [note for note in notes if is_alt_name(note)]
    # remove or, and split the alt name by ,
    if len(alt_names) != 0:
        alt_names = alt_names[0].replace("or,", "").split(",")
        alt_names = [alt_name.strip() for alt_name in alt_names]
        return alt_names
    return None


def is_category(line: str):
    """Check if the line is a category"""
    # categories are in all uppercase and may optionally have 's in lowercase
    return line == line.upper() or line == line.upper().replace("'S", "'s")


def is_subcategory(line: str):
    """Check if the line is a subcategory"""
    # subcategories are surrounded by []
    return line[0] == "[" and line[-1] == "]"


def get_subcategory(line: str):
    """Get the subcategory from the line"""
    if is_subcategory(line):
        return line[1:-1].strip()
    return None


def get_categories(lines: List[str]):
    """Get all the categories in the recommendations"""
    # categories are in all uppercase
    categories = [line for line in lines if is_category(line)]
    return set(categories)


def group_by_category(lines: List[str], categories: Set[str]):
    """Group the lines by category"""
    grouped_lines = {}
    current_category = None
    for line in lines:
        if line in categories:
            grouped_lines[line] = []
            current_category = line
        else:
            grouped_lines[current_category].append(line)
    return grouped_lines


def preprocess(grouped_lines: Dict[str, List[str]]):
    """Extract the data from the grouped lines"""
    data = []
    for category, lines in grouped_lines.items():
        subcategory = NO_SUBCATEGORY
        for line in lines:
            if is_subcategory(line):
                subcategory = get_subcategory(line)
            else:
                rating = get_rating_from_title(line)
                notes = get_notes_from_title(line)
                alt_names = get_alt_names_from_notes(notes) if notes else None
                # remove the alt_names from the notes if the notes contain alt_names
                if notes and alt_names:
                    notes = [note for note in notes if not is_alt_name(note)]
                # remove the rating and notes from the title
                line = re.sub(r"\(.*?\)", "", line).strip()
                data.append(
                    {
                        "category": category,
                        "subcategory": subcategory,
                        "title": line,
                        "rating": rating,
                        "notes": notes,
                        "alt_names": alt_names,
                    }
                )
    return data


def group_by_title(data_list: List[Dict[str, str]]):
    """Group the data by title"""
    grouped_data = {}

    for entry in data_list:
        title = entry["title"]

        # Initialize the title's entry if not present
        if title not in grouped_data:
            grouped_data[title] = {
                "categories": [],
                "subcategories": [],
                "rating": [],
                "notes": [],
                "alt_names": [],
            }

        # Update categories, subcategories, notes, and alt_names
        grouped_data[title]["categories"].append(entry["category"])
        if entry["subcategory"]:
            grouped_data[title]["subcategories"].append(entry["subcategory"])
        if entry["notes"]:
            grouped_data[title]["notes"].extend(entry["notes"])
        if entry["alt_names"]:
            grouped_data[title]["alt_names"].extend(entry["alt_names"])

        # Handle ratings
        if entry["rating"]:
            grouped_data[title]["rating"].append(entry["rating"])

    # Compute average rating for each title, if applicable
    for title, values in grouped_data.items():
        if values["rating"]:
            avg_rating = sum(values["rating"]) / len(values["rating"])
            grouped_data[title]["rating"] = avg_rating
        else:
            grouped_data[title]["rating"] = None

        # Ensure unique values for categories, subcategories, notes, and alt_names
        grouped_data[title]["categories"] = list(set(values["categories"]))
        grouped_data[title]["subcategories"] = list(set(values["subcategories"]))
        grouped_data[title]["notes"] = list(set(values["notes"]))
        grouped_data[title]["alt_names"] = list(set(values["alt_names"]))

    return grouped_data


def load_raw_data():
    with open(RAW_REDDIT_RECOMMENDATIONS_TXT_FILE_PATH, "r") as f:
        lines = f.readlines()
        return [line.strip() for line in lines if line.strip() != ""]


def run():
    print(
        f"Loading raw reddit recommendations from: {RAW_REDDIT_RECOMMENDATIONS_TXT_FILE_PATH}"
    )
    lines = load_raw_data()
    print("Processing data...")
    categories_set = get_categories(lines)
    grouped_lines = group_by_category(lines, categories_set)
    grouped_lines_with_subcategories = group_by_title(preprocess(grouped_lines))
    print("Done!")
    with open(RAW_REDDIT_RECOMMENDATIONS_JSON_FILE_PATH, "w") as f:
        print(f"Writing parsed data to: {RAW_REDDIT_RECOMMENDATIONS_JSON_FILE_PATH}...")
        json.dump(grouped_lines_with_subcategories, f, indent=4)


if __name__ == "__main__":
    run()
