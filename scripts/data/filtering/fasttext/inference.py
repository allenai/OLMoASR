from typing import Iterable, List, Tuple
import re
import string
from open_whisper.utils import TranscriptReader

from dolma.core.data_types import TextSlice
from dolma.core.ft_tagger import BaseFastTextTagger, Prediction
from dolma import add_tagger

@add_tagger("ow-tedlium-quality")
class OWTedliumQualityClassifier(BaseFastTextTagger):
    MODEL_PATH = "/mnt/raid0/tedlium.bin"
    MAX_CHAR_LEN = 376

    def __init__(self):
        self.max_char_len = self.MAX_CHAR_LEN
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def modify_text(self, text):
        pattern_brackets = (
            r"[ ]*\[(?![Mm][Uu][Ss][Ii][Cc]\])([A-Z][a-zA-Z]*(?: [A-Z][a-zA-Z]*)*)\][ ]*"
        )
        pattern_parentheses = r"[ ]*\(.*?\)[ ]*"
        pattern_colon = r"[ ]*(?:[A-Z][a-zA-Z]*[ ])+:[ ]*"
        specific_strings = r"[ ]*(?:&nbsp;|&amp;|&lt;|&gt;|=|\.{3})+[ ]*"
        primary_pattern = (
            f"{pattern_brackets}|{pattern_parentheses}|{pattern_colon}|{specific_strings}"
        )
        brackets_pattern_capture = r"\[([a-z]+(?: [a-z]+)*)\]"

        text = re.sub(primary_pattern, " ", text)
        text = re.sub(brackets_pattern_capture, r"\1", text)

        return text

    def preprocess(self, transcript_string: str) -> List[Tuple[str, Tuple[int, int]]]:
        reader = TranscriptReader(
            file_path=None,
            transcript_string=transcript_string,
            ext="vtt" if transcript_string.startswith("WEBVTT") else "srt",
        )
        t_dict, *_ = reader.read()
        text = reader.extract_text(t_dict)
        text = self.modify_text(text)
        text = text.strip()
        text = text.lower()
        punctuation_to_remove = string.punctuation.replace("'", "") + "“" + "”"
        text = text.translate(str.maketrans("", "", punctuation_to_remove))
        text = re.sub(r"\s*\n\s*", " ", text)
        text = text[:self.max_char_len] if self.max_char_len else text
        return text

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        text = self.preprocess(text_slice.text)
        pred = self.classifier.predict(text, k=-1)
        
        # Extract the predicted label and its probability
        (pred_label, pred_prob) = pred
        pred_label = pred_label[0]
        probability_score = pred_prob[0]

        if pred_label == "__label__negative":
            probability_score = 1 - probability_score

        label = pred_label.replace("__label__", "").replace("positive", "score").replace("negative", "score")

        return [Prediction(label=label, score=probability_score)]
    
@add_tagger("ow-commonvoice-quality")
class OWCVQualityClassifier(BaseFastTextTagger):
    MODEL_PATH = "/mnt/raid0/commonvoice.bin"
    MAX_CHAR_LEN = 211

    def __init__(self):
        self.max_char_len = self.MAX_CHAR_LEN
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def modify_text(self, text):
        pattern_brackets = (
            r"[ ]*\[(?![Mm][Uu][Ss][Ii][Cc]\])([A-Z][a-zA-Z]*(?: [A-Z][a-zA-Z]*)*)\][ ]*"
        )
        pattern_parentheses = r"[ ]*\(.*?\)[ ]*"
        pattern_colon = r"[ ]*(?:[A-Z][a-zA-Z]*[ ])+:[ ]*"
        specific_strings = r"[ ]*(?:&nbsp;|&amp;|&lt;|&gt;|=|\.{3})+[ ]*"
        primary_pattern = (
            f"{pattern_brackets}|{pattern_parentheses}|{pattern_colon}|{specific_strings}"
        )
        brackets_pattern_capture = r"\[([a-z]+(?: [a-z]+)*)\]"

        text = re.sub(primary_pattern, " ", text)
        text = re.sub(brackets_pattern_capture, r"\1", text)

        return text

    def preprocess(self, transcript_string: str) -> List[Tuple[str, Tuple[int, int]]]:
        reader = TranscriptReader(
            file_path=None,
            transcript_string=transcript_string,
            ext="vtt" if transcript_string.startswith("WEBVTT") else "srt",
        )
        t_dict, *_ = reader.read()
        text = reader.extract_text(t_dict)
        text = self.modify_text(text)
        text = text.strip()
        text = text.lower()
        punctuation_to_remove = string.punctuation.replace("'", "") + "“" + "”"
        text = text.translate(str.maketrans("", "", punctuation_to_remove))
        text = re.sub(r"\s*\n\s*", " ", text)
        text = text[:self.max_char_len] if self.max_char_len else text
        return text

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        text = self.preprocess(text_slice.text)
        pred = self.classifier.predict(text, k=-1)
        
        # Extract the predicted label and its probability
        (pred_label, pred_prob) = pred
        pred_label = pred_label[0]
        probability_score = pred_prob[0]

        if pred_label == "__label__negative":
            probability_score = 1 - probability_score

        label = pred_label.replace("__label__", "").replace("positive", "score").replace("negative", "score")

        return [Prediction(label=label, score=probability_score)]