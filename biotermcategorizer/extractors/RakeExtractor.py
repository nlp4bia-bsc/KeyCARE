import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
from .Extractor import Extractor

class RakeExtractor(Extractor):
    def __init__(self, language, max_tokens):
        """
        Initializes a RakeExtractor object with specified parameters and the Rake object itself.

        Parameters:
        language (str): Language for text processing.
        max_tokens (int): Maximum number of tokens for extracted terms.

        Returns:
        None
        """
        super().__init__(language, max_tokens)
        self.stopwords = nltk.corpus.stopwords.words(language)
        self.extractor = Rake(stopwords=self.stopwords, language=language)

    def extract_terms(self, text):
        """
        Extracts terms using the RAKE algorithm and the specified parameters.

        Parameters:
        text (str): Input text for term extraction.

        Returns:
        list: List of extracted terms with scores.
        """
        filtered_kwargs = {key: value for key, value in self.kwargs.items() if key in self.extractor.extract_keywords_from_text.__code__.co_varnames}
        self.extractor.extract_keywords_from_text(text, **filtered_kwargs)
        terms = self.extractor.get_ranked_phrases_with_scores()
        terms = [(kw,score) for score,kw in terms if (len(word_tokenize(kw, language=self.language)) <= self.max_tokens)]
        return terms