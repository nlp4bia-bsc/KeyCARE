from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import os
import nltk
nltk.download('stopwords')
from keyphrase_vectorizers import KeyphraseCountVectorizer
from .Extractor import Extractor

class KeyBertExtractor(Extractor):
    def __init__(self, language, max_tokens, pos, pos_pattern, top_n):
        """
        Initializes the KeyBertExtractor object with specified parameters and the KeyBERT object itself.

        Parameters:
        language (str): Language for keyword extraction.
        max_tokens (int): Maximum number of tokens in extracted keywords.
        pos (bool): Wether to use Part of Speech sequences.
        pos_pattern (str): The Part of Speech regex pattern.
        top_n (int): number of keywords to be extracted.
        """
        super().__init__(language, max_tokens, top_n)
        sentence_model = self.initialize_sentence_model()
        self.extractor = KeyBERT(model=sentence_model)
        self.pos = pos
        self.pos_pattern = pos_pattern
        self.top_n = top_n

    def initialize_sentence_model(self):
        """
        Initializes the sentence embedding model.

        Parameters:
        None.

        Returns:
        SentenceTransformer: Initialized sentence embedding model.
        """
        path = 'BSC-NLP4BIA/SapBERT-parents-from-roberta-base-biomedical-clinical-es'
        sentence_model = SentenceTransformer(path)
        return sentence_model

    def extract_terms(self, text):
        """
        Extracts keywords from the given text using KeyBERT (with PoS or with n_gram_range).

        Parameters:
        text (str): Input text for keyword extraction.

        Returns:
        list: List of extracted keywords and their scores, sorted by score in descending order.
        """
        self.select_top_n(text)
        filtered_kwargs = {key: value for key, value in self.kwargs.items() if key in self.extractor.extract_keywords.__code__.co_varnames}
        specified_params = ['keyphrase_ngram_range', 'stop_words', 'use_mmr']
        for param in specified_params:
            if param in filtered_kwargs:
                del filtered_kwargs[param]
                print("Warning. The parameter " + param + " can not be changed from its given value.")
        keywords_all = list()
        if self.pos:
            keywords_all = self.extractor.extract_keywords(text, vectorizer=KeyphraseCountVectorizer(spacy_pipeline="es_core_news_sm", stop_words='spanish', pos_pattern=self.pos_pattern), top_n=self.top_n, **filtered_kwargs)
        else:
            for i in range(self.max_tokens):
                keywords_i = self.extractor.extract_keywords(text, keyphrase_ngram_range=(1, i+1), stop_words=None, use_mmr=True, top_n=self.top_n, **filtered_kwargs)
                keywords_all.extend(keywords_i)
        terms = sorted(keywords_all, key=lambda x: x[1], reverse=True)
        return terms
