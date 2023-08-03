class RakeExtractor(Extractor):
    def __init__(self, stopwords, language):
        super().__init__()
        self.stopwords = stopwords
        self.language = language

    def extract_terms(self, text):
        # Lógica para la extracción de términos mediante string matching
        terms = []  # Ejemplo de términos encontrados
        return terms
