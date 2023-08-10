import os, re
from nltk.tokenize import word_tokenize

class TermExtractor:
    def __init__(self, extraction_methods="rake", language="spanish", max_tokens=3): #maybe fer que max_tokens s hagi de posar mes tard??
        #he puesto valores default
        self.extraction_methods = extraction_methods
        self.extractors = self.initialize_keyword_extractors(language, max_tokens)

    def __call__(self, text):
        terms = self.extract_terms(text)
        return terms
        
    def initialize_keyword_extractors(self, language, max_tokens):
        keyword_extractors = {} #esto esta hecho para mas de un extractor a la vez???
        
        if 'rake' in self.extraction_methods:
            keyword_extractors["rake"] = RakeExtractor(language, max_tokens)
        
        if 'yake' in self.extraction_methods:
            keyword_extractors["yake"] = YakeExtractor(language, max_tokens)
        
        if 'textrank' in self.extraction_methods:
            keyword_extractors["textrank"] = TextRankExtractor(language, max_tokens)

        if not keyword_extractors:
            raise ValueError("No extraction method called {}".format(self.extraction_methods))
        
        return keyword_extractors

    def extract_terms(self, text):
        if (len(self.extractors) == 1):
            terms = self.extractors[self.extraction_methods].extract_terms_without_overlaps(text) #solo funciona si solo he puesto un method
        else:
            all_terms = []
            for extractor in self.extractors.values():
                all_terms.extend(extractor.extract_terms_with_span(text))
            terms = self.extractors[self.extraction_methods[0]].rmv_overlaps(all_terms)
        return terms
    