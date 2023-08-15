import nltk
nltk.download('stopwords')
nltk.download('punkt')
from rake_nltk import Rake

class RakeExtractor(Extractor):
    def __init__(self, language, max_tokens):
        super().__init__(language, max_tokens)
        self.stopwords = nltk.corpus.stopwords.words(language)
        self.extractor = Rake(stopwords=self.stopwords, language=language)

    def extract_terms(self, text):
        self.extractor.extract_keywords_from_text(text)
        terms = self.extractor.get_ranked_phrases_with_scores()
        terms = [(kw,score) for score,kw in terms if (len(word_tokenize(kw, language=self.language)) <= self.max_tokens)]
        return terms