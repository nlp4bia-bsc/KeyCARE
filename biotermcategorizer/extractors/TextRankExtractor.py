import spacy, pytextrank

class TextRankExtractor(Extractor):
    def __init__(self, language, max_tokens):
        super().__init__(language, max_tokens)
        if language=='spanish':
            self.extractor = spacy.load("es_core_news_sm")
            self.extractor.add_pipe("textrank")
        else:
            raise ValueError("Expected spanish language. Other languages not recognised")

    def extract_terms(self, text):
        doc = self.extractor(text)
        terms = []
        for phrase in doc._.phrases:
          if (len(word_tokenize(phrase.text)) <= self.max_tokens):
            terms.append(phrase.text)
        return terms