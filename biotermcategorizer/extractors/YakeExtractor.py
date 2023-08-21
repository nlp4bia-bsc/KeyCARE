import yake
from nltk.tokenize import word_tokenize
from .Extractor import Extractor

class YakeExtractor(Extractor):
    def __init__(self, language, max_tokens):
        """
        Initializes a YakeExtractor object with specified parameters.

        Parameters:
        language (str): Language for text processing.
        max_tokens (int): Maximum number of tokens for extracted terms.

        Returns:
        None
        """
        super().__init__(language, max_tokens)
        if language == 'spanish':
            self.extractor = yake.KeywordExtractor()  # here you might want to set additional parameters like top=70
        else:
            raise ValueError("Expected spanish language. Other languages not recognized")

    def extract_terms(self, text):
        """
        Extracts terms using the YAKE algorithm.

        Parameters:
        text (str): Input text for term extraction.

        Returns:
        list: List of extracted terms with scores.
        """
        keywords = self.extractor.extract_keywords(text)
        terms = [(kw, score) for kw, score in keywords if (len(word_tokenize(kw)) <= self.max_tokens)]
        return terms
