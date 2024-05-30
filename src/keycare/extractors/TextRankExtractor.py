import spacy, pytextrank
from nltk.tokenize import word_tokenize
from .Extractor import Extractor

class TextRankExtractor(Extractor):
    def __init__(self, language, max_tokens, top_n):
        """
        Initializes a TextRankExtractor object with specified parameters.

        Parameters:
        language (str): Language for text processing.
        max_tokens (int): Maximum number of tokens for extracted terms.
        top_n (int): number of keywords to be extracted.
        """
        super().__init__(language, max_tokens, top_n)
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
        self.select_top_n(text)
        filtered_kwargs = {key: value for key, value in self.kwargs.items() if key in self.extractor.__call__.__code__.co_varnames}
        doc = self.extractor(text, **filtered_kwargs)
        terms = []
        for phrase in doc._.phrases:
          if (len(word_tokenize(phrase.text)) <= self.max_tokens):
            terms.append((phrase.text, phrase.rank))
        sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)
        try:
            return sorted_terms[:self.top_n]
        except IndexError:
            return sorted_terms