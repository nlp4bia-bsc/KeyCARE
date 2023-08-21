import spacy
import pytextrank
from nltk.tokenize import word_tokenize
from .Extractor import Extractor

class TextRankExtractor(Extractor):
    def __init__(self, language, max_tokens):
        """
        Initializes a TextRankExtractor object with specified parameters.

        Parameters:
        language (str): Language for text processing.
        max_tokens (int): Maximum number of tokens for extracted terms.

        Returns:
        None
        """
        super().__init__(language, max_tokens)
        if language == 'spanish':
            self.extractor = spacy.load("es_core_news_sm")
            self.extractor.add_pipe("textrank")
        else:
            raise ValueError("Expected spanish language. Other languages not recognized")

    def extract_terms(self, text):
        """
        Extracts terms using the TextRank algorithm.

        Parameters:
        text (str): Input text for term extraction.

        Returns:
        list: List of extracted terms with scores/ranks.
        """
        doc = self.extractor(text)
        terms = []
        for phrase in doc._.phrases:
            if (len(word_tokenize(phrase.text)) <= self.max_tokens):
                terms.append((phrase.text, phrase.rank))
        return terms
