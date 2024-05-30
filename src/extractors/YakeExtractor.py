import yake
from nltk.tokenize import word_tokenize
from .Extractor import Extractor

class YakeExtractor(Extractor):
    def __init__(self, language, max_tokens, top_n):
        """
        Initializes a YakeExtractor object with specified parameters.

        Parameters:
        language (str): Language for text processing.
        max_tokens (int): Maximum number of tokens for extracted terms.
        top_n (int): number of keywords to be extracted.
        """
        super().__init__(language, max_tokens, top_n)
        if language == 'spanish':
            self.extractor = yake.KeywordExtractor() 
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
        self.select_top_n(text)
        filtered_kwargs = {key: value for key, value in self.kwargs.items() if key in self.extractor.extract_keywords.__code__.co_varnames}
        keywords = self.extractor.extract_keywords(text, **filtered_kwargs)
        terms = [(kw,score) for kw, score in keywords if (len(word_tokenize(kw)) <= self.max_tokens)]
        sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)
        try:
            return sorted_terms[:self.top_n]
        except IndexError:
            return sorted_terms