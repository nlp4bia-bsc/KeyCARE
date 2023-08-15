import os, re
from nltk.tokenize import word_tokenize

class TermExtractor:
    def __init__(self, extraction_methods="rake", language="spanish", max_tokens=3, join=False): #maybe fer que max_tokens s hagi de posar mes tard??
        self.extraction_methods = extraction_methods
        self.extractors = self.initialize_keyword_extractors(language, max_tokens)
        self.keywords = None
        self.join = join

    def __call__(self, text):
        terms = self.extract_terms(text, self.join)
        return terms
        
    def initialize_keyword_extractors(self, language, max_tokens):
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

    def extract_terms(self, text, join):
        try:
            all_terms = []
            for key, extractor in self.extractors.items():
                terms = extractor.extract_terms_without_overlaps(text)
                terms = [term + (key,) for term in terms]
                all_terms.extend(terms)
            if join:
                all_terms = list(self.extractors.values())[0].rmv_overlaps(all_terms)
            self.keywords = [Keyword(text=i[0], method=i[4], ini=i[1], fin=i[2], score=i[3]) for i in all_terms]
        except:
            raise AttributeError("A list of extractors must be provided")
    