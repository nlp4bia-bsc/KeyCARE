class KeyBertExtractor(Extractor):
    def __init__(self, language, max_tokens):
        """
        Initializes the KeyBertExtractor object with specified parameters and the KeyBERT object itself.

        Parameters:
        language (str): Language for keyword extraction.
        max_tokens (int): Maximum number of tokens in extracted keywords.
        """
        super().__init__(language, max_tokens)
        sentence_model = self.initialize_sentence_model()
        self.extractor = KeyBERT(model=sentence_model)

    def initialize_sentence_model(self):
        """
        Initializes the sentence embedding model.

        Parameters:
        None.

        Returns:
        SentenceTransformer: Initialized sentence embedding model.
        """
        path = '/mnt/c/Users/Sergi/Desktop/BSC/spanish_sapbert_models/sapbert_15_parents_1epoch'
        sentence_model = SentenceTransformer(path)
        return sentence_model

    def extract_terms(self, text):
        """
        Extracts keywords from the given text using KeyBERT.

        Parameters:
        text (str): Input text for keyword extraction.

        Returns:
        list: List of extracted keywords and their scores, sorted by score in descending order.
        """
        filtered_kwargs = {key: value for key, value in self.kwargs.items() if key in self.extractor.extract_keywords.__code__.co_varnames}
        specified_params = ['keyphrase_ngram_range', 'stop_words', 'use_mmr']
        for param in specified_params:
            if param in filtered_kwargs:
                del filtered_kwargs[param]
                print("Warning. The parameter " + param + " can not be changed from its given value.")
        keywords_all = list()
        for i in range(self.max_tokens):
            keywords_i = self.extractor.extract_keywords(text, keyphrase_ngram_range=(1, i+1), stop_words=None, use_mmr=True, **filtered_kwargs)
            keywords_all.extend(keywords_i)
        terms = sorted(keywords_all, key=lambda x: x[1], reverse=True)
        return terms
