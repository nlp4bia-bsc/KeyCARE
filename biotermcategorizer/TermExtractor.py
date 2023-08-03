class TermExtractor:
    def __init__(self, extraction_methods):
        self.extraction_methods = extraction_methods
        self.extractors = self.initialize_keyword_extractors()

    def __call__(self, text):
        self.extract_terms(text)
        
    def initialize_keyword_extractors(self):
        keyword_extractors = {}

        if 'rake' in self.extraction_methods:
            keyword_extractors["rake"] = "OBJETO YAKE EXTRACTOR" #YakeExtractor()
            
        if 'textrank' in self.extraction_methods:
            keyword_extractors["textrank"] = "OBJETO TEXTRANK EXTRACTOR"
        
        return keyword_extractors

    def extract_terms(self, text):
        print(self.extractors)
        print(text)
    