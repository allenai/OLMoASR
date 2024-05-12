from typing import Optional
from langdetect import detect

def standardize_dialects(s: str) -> str:
    """In the manual_caption_languages column in a metadata file, standardize dialects to their base language

    Takes in a string that can contain a single language code or multiple language codes separated by commas.
    If the function detects a language code with a dialect, it will transform it to the base language code.

    Args:
        s: String containing language codes

    Returns:
        A string with all the base language codes
    """
    words = s.split(",")
    transformed_words = [word.split("-")[0] if "-" in word else word for word in words]
    return ",".join(transformed_words)


# for rough filtering of english-only videos
def detect_en(row) -> Optional[int]:
    """Detect whether title is in English or not

    Args:
        row: Tuple containing index and data

    Returns:
        Index of the row if the title is in English, otherwise None
    """
    idx, data = row
    try:
        if detect(data["title"]) == "en":
            return idx
        else:
            return None
    except Exception:
        pass