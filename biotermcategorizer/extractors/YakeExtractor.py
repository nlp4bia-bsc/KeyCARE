import yake

class YakeExtractor(Extractor):
    def __init__(self, language, max_tokens):
        super().__init__(language, max_tokens)
        if language=='spanish':
            self.extractor = yake.KeywordExtractor() #aqui habria que poner top=70 por ejemplo
        else:
            raise ValueError("Expected spanish language. Other languages not recognised")

    def extract_terms(self, text):
        keywords = self.extractor.extract_keywords(text)
        terms = [(kw,score) for kw, score in keywords if (len(word_tokenize(kw)) <= self.max_tokens)]
        return terms