import os
import re
from nltk.tokenize import word_tokenize
from extractors.RakeExtractor import RakeExtractor
from extractors.YakeExtractor import YakeExtractor
from extractors.TextRankExtractor import TextRankExtractor
from utils.data_structures import Keyword

class TermExtractor:
    def __init__(self, extraction_methods=["textrank"], language="spanish", max_tokens=3, join=False, postprocess=True):
        """
        Initializes a TermExtractor object with specified parameters.

        Parameters:
        extraction_methods (list): List of extraction methods to use.
        language (str): Language for text processing.
        max_tokens (int): Maximum number of tokens for extracted terms.
        join (bool): Whether to join terms obtained with different methods and to remove overlaps among them.

        Returns:
        None
        """
        self.extraction_methods = extraction_methods
        self.extractors = self.initialize_keyword_extractors(language, max_tokens)
        self.keywords = None
        self.join = join
        self.postprocess = postprocess

    def __call__(self, text):
        """
        Extracts terms from the given text using initialized extractors.

        Parameters:
        text (str): Input text for term extraction.

        Returns:
        list: List of extracted terms.
        """
        terms = self.extract_terms(text, self.join, self.postprocess)
        return terms

    def initialize_keyword_extractors(self, language, max_tokens):
        """
        Initializes keyword extractors based on selected extraction methods.

        Parameters:
        language (str): Language for text processing.
        max_tokens (int): Maximum number of tokens for extracted terms.

        Returns:
        dict: Dictionary containing initialized keyword extractors.
        """
        keyword_extractors = {}

        if 'rake' in self.extraction_methods:
            keyword_extractors["rake"] = RakeExtractor(language, max_tokens)

        if 'yake' in self.extraction_methods:
            keyword_extractors["yake"] = YakeExtractor(language, max_tokens)

        if 'textrank' in self.extraction_methods:
            keyword_extractors["textrank"] = TextRankExtractor(language, max_tokens)

        if not keyword_extractors:
            raise ValueError("No extraction method called {}".format(self.extraction_methods))

        return keyword_extractors

    def extract_terms(self, text, join, postprocess):
        """
        Extracts terms from the given text using initialized extractors.

        Parameters:
        text (str): Input text for term extraction.
        join (bool): Whether to join terms obtained with different methods and to remove overlaps among them.

        Returns:
        list: List of extracted terms.
        """
        try:
            all_terms = []
            for key, extractor in self.extractors.items():
                terms = extractor.extract_terms_without_overlaps(text)
                terms = [term + (key,) for term in terms]
                all_terms.extend(terms)
            if join:
                all_terms = list(self.extractors.values())[0].rmv_overlaps(all_terms)
            if postprocess:
                all_terms = list(self.extractors.values())[0].postprocess_terms(all_terms)
            self.keywords = [Keyword(text=i[0], method=i[4], ini=i[1], fin=i[2], score=i[3]) for i in all_terms]
        except:
            raise AttributeError("A list of extractors must be provided")

    