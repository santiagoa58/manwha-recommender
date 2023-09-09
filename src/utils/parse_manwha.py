import json
from typing import Dict
from bs4 import BeautifulSoup
import re
from src.utils.constants import UNKNOWN


def parse_number(str_number: str):
    assert isinstance(str_number, str)
    # Check if the string is an integer or a float
    try:
        # First, try to convert to int
        return int(str_number)
    except ValueError:
        # If not an int, try to convert to float
        return float(str_number)


def is_str_number(str_number: str):
    try:
        parse_number(str_number)
        # no error, so it is a number
        return True
    except Exception as _e:
        # error, so it is not a number
        return False


# replace unicode characters with their closest ASCII equivalent
def unidecode(text: str):
    return "".join(char for char in text if 32 <= ord(char) <= 126)


def clean_text(text: str | None):
    if not text:
        return UNKNOWN
    text = unidecode(text)
    # remove non-printable characters using char for char in text if char in cleaned_text
    cleaned_text = [char for char in text if char in text]
    return "".join(cleaned_text)


def clean_value(value: int | float | str | None):
    if not value and value != 0:
        return UNKNOWN
    if isinstance(value, (int, float)):
        return value
    try:
        if is_str_number(value):
            return parse_number(value)
    except Exception as e:
        print(f"Error: {value} could not be converted to a number\n{e}")
        return UNKNOWN
    return clean_text(value)


def extract_manwha_info(soup: BeautifulSoup):
    def get_name():
        # the manwha name has the class "theme-font"
        # ex: <h5 class='theme-font'>Solo Leveling</h5>
        manwha_name_element = soup.select_one(".theme-font")
        return clean_text(manwha_name_element.text) if manwha_name_element else UNKNOWN

    def get_alt_name():
        # the manwha alt name has the class "tooltip-alt";
        # ex: <h6 class='theme-font tooltip-alt'>Alt titles: Na Honjaman Level-Up, Only I Level Up</h6>
        manwha_alt_name_element = soup.select_one(".tooltip-alt")
        return (
            clean_text(
                manwha_alt_name_element.text.replace("Alt title: ", "").replace(
                    "Alt titles: ", ""
                )
            )
            if manwha_alt_name_element
            else UNKNOWN
        )

    def get_rating():
        # the rating has the class "ttRating"
        # ex: <li><div class='ttRating'>4.7</div></li>
        rating_element = soup.select_one(".ttRating")
        return clean_value(rating_element.text) if rating_element else UNKNOWN

    def get_chapters():
        # the chapters and volumes have the class "iconVol"
        # ex: <li class='iconVol'>Ch: 200</li>
        chapter_volume_element = soup.select_one(".iconVol")
        if chapter_volume_element:
            chapters_match = re.search(r"Ch: (\d+\+?)", chapter_volume_element.text)
            volumes_match = re.search(r"Vol: (\d+\+?)", chapter_volume_element.text)
            chapters = chapters_match.group(1) if chapters_match else None
            volumes = volumes_match.group(1) if volumes_match else None
            return {"chapters": clean_value(chapters), "volumes": clean_value(volumes)}
        return UNKNOWN

    def get_publisher():
        # the publisher is the child of an element with the class "entryBar" and the second li element
        # ex: <li>Kakao Page</li>
        publisher_element = soup.select_one(".entryBar li:nth-child(2)")
        if (
            publisher_element
            and publisher_element.text
            and not publisher_element.attrs.get("class")
        ):
            return clean_text(publisher_element.text)
        return UNKNOWN

    def get_years():
        # the years the manwha was active has the class "iconYear"
        # <li class='iconYear'>2018 - 2023</li>
        years_active_element = soup.select_one(".iconYear")
        return (
            clean_value(years_active_element.text) if years_active_element else UNKNOWN
        )

    def get_description():
        # the manwha description is the child of an element with the class "pure-2-3"
        # ex: <div class='pure-2-3'><p>E-class hunter Jinwoo Sung is the weakest of them all...</p></div>
        description_element = soup.select_one(".pure-2-3 p")
        return clean_text(description_element.text) if description_element else UNKNOWN

    def get_source():
        # the manwha source is the child of an element with the class "tooltip notes"
        # ex: <div class='tooltip notes'style='padding-top: 1em;'><p>Source:&nbsp;Yen Press</p></div>
        manwha_source_element = soup.select_one(".tooltip.notes p")
        return (
            clean_text(manwha_source_element.text.replace("Source: ", ""))
            if manwha_source_element
            else UNKNOWN
        )

    def get_image_url():
        # the manwha image url is the img child of an element with the class "pure-1-3"
        # ex: <div class='pure-1-3'><img alt='Solo Leveling' src='https://cdn.anime-planet.com/manga/primary/solo-leveling-1-190x273.jpg?t=1625919488' /></div>
        img = soup.select_one(".pure-1-3 img")
        return clean_value(img["src"]) if img else UNKNOWN

    def get_tags():
        # the manwha tags are the li children of an element with the class "tags"
        # ex: <div class='tags'><h4>Tags</h4><ul><li>Action</li><li>Adventure</li>...</ul></div>
        tag_elements = soup.select(".tags ul li")
        cleaned_tags = [
            clean_text(tag_element.text) for tag_element in tag_elements if tag_element
        ]
        return [tag for tag in cleaned_tags if tag != "Unknown"]

    return {
        "name": get_name(),
        "altName": get_alt_name(),
        "rating": get_rating(),
        "chapters": get_chapters(),
        "publisher": get_publisher(),
        "years": get_years(),
        "description": get_description(),
        "source": get_source(),
        "imageURL": get_image_url(),
        "tags": get_tags(),
    }


def parse_manwha(data: Dict[str, str]):
    assert isinstance(data, dict)
    manwhas = []
    for id, html_string in data.items():
        soup = BeautifulSoup(html_string, "html.parser")
        manwha_info = extract_manwha_info(soup)
        # add the id to the manwha_info
        manwha_info["id"] = id
        manwhas.append(manwha_info)
    return manwhas


def load_manwha():
    with open("./data/rawManwhas.json") as raw_manwha:
        print("Loading manwhas...")
        data = json.load(raw_manwha)
        print("manwhas loaded!")
        print("parsing manwhas...")
        manwhas = parse_manwha(data)
        print("manwhas parsed!")
        return manwhas


def write_manwha_to_file(manwhas):
    with open("./data/cleanedManwhas.json", "w") as cleaned_manwha:
        print("Writing manwhas to cleanedManwhas.json...")
        json.dump(manwhas, cleaned_manwha, indent=2)
    print("Finished writing manwhas to cleanedManwhas.json")


def run():
    try:
        manwhas = load_manwha()
        write_manwha_to_file(manwhas)
        print("Done!")
    except Exception as e:
        print(f"Error parsing manwhas\n")
        raise e


if __name__ == "__main__":
    run()
