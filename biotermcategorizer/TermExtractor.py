import os
import re
from nltk.tokenize import word_tokenize
from extractors.RakeExtractor import RakeExtractor
from extractors.YakeExtractor import YakeExtractor
from extractors.TextRankExtractor import TextRankExtractor
from utils.data_structures import Keyword
from categorizers.Clustering import Clustering
from categorizers.SetFitClassifier import SetFitClassifier
from categorizers.StandardClassifier import StandardClassifier

class TermExtractor:
    def __init__(self, extraction_methods=["textrank"], categorizer_method="setfit", language="spanish", max_tokens=3, join=False, postprocess=True, parents=False, n=1, threshold=0.5):
        """
        Initializes a TermExtractor object with specified parameters.

        Parameters:
        extraction_methods (list): List of extraction methods to use.
        categorizer_method (str): Method for categorizing extracted terms.
        language (str): Language for text processing.
        max_tokens (int): Maximum number of tokens for extracted terms.
        join (bool): Whether to join terms obtained with different methods and to remove overlaps among them.
        postprocess (bool): Whether to perform post-processing on extracted keywords.
        parents (bool): Whether to use models with parents in SetFitClassifier.
        n (int): Maximum number of predicted labels to consider in SetFitClassifier.
        threshold (float): Threshold for label filtering in SetFitClassifier.
        
        Returns:
        None
        """
        self.extraction_methods = extraction_methods
        self.extractors = self.initialize_keyword_extractors(language, max_tokens)
        self.categorizer_method = categorizer_method
        self.categorizer = self.initialize_categorizers(parents, n, threshold)
        self.join = join
        self.postprocess = postprocess

    #def __init__(self, **kwargs):
    #    for key, value in kwargs.items():
    #		setattr(self, key, value)
    
    def __call__(self, text):
        """
        Extracts terms from the given text using initialized extractors and categorize them using initialized categorizers.

        Parameters:
        text (str): Input text for term extraction.

        Returns:
        None
        """
        self.extract_terms(text, self.join, self.postprocess)
        self.categorize_terms()

    def initialize_keyword_extractors(self, language, max_tokens):
        """
        Initializes keyword extractors based on selected extraction methods.

        Parameters:
        language (str): Language for text processing.
        max_tokens (int): Maximum number of tokens for extracted terms.

        Returns:
        dict: Dictionary containing initialized keyword extractors.
        """
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

    def extract_terms(self, text, join, postprocess):
        """
        Extracts terms from the given text using initialized extractors.

        Parameters:
        text (str): Input text for term extraction.
        join (bool): Whether to join terms obtained with different methods and to remove overlaps among them.

        Returns:
        list: List of extracted terms.
        """
        try:
            all_terms = []
            for key, extractor in self.extractors.items():
                terms = extractor.extract_terms_without_overlaps(text)
                terms = [term + (key,) for term in terms]
                all_terms.extend(terms)
            if join:
                all_terms = list(self.extractors.values())[0].rmv_overlaps(all_terms)
            if postprocess:
                all_terms = list(self.extractors.values())[0].postprocess_terms(all_terms)
            self.keywords = [Keyword(text=i[0], method=i[4], ini=i[1], fin=i[2], score=i[3]) for i in all_terms]
        except:
            raise AttributeError("A list of extractors must be provided")

    def initialize_categorizers(self, parents, n, threshold):
        """
        Initializes the categorizer based on the selected method.

        Parameters:
        parents (bool): Whether to use parent models in SetFitClassifier.
        n (int): Maximum number of predicted labels to consider in SetFitClassifier.
        threshold (float): Threshold for label filtering in SetFitClassifier.

        Returns:
        Categorizer: Initialized categorizer object.
        """
        if 'standard' == self.categorizer_method:
            categorizer = StandardClassifier()
            
        elif 'setfit' == self.categorizer_method:
            categorizer = SetFitClassifier(parents, n, threshold)

        elif 'clustering' == self.categorizer_method:
            categorizer = Clustering()

        else:
            raise ValueError("No categorizer method called {}".format(self.categorizer_method))
        return categorizer

    def categorize_terms(self):
        """
        Categorizes the extracted terms (keyword objects) using the selected categorizer method.
        """
        try:
            if self.categorizer_method == 'clustering':
                clusters = self.categorizer.compute_clusters(self.keywords)
                for kw in self.keywords:
                    kw.label = clusters[kw.text]
            else:
                for kw in self.keywords:
                    kw.label = self.categorizer.compute_predictions(kw.text)
        except:
            raise AttributeError("A categorizer method must be provided")

    def train_classifier(self, trainset, testset, mcm=False, classification_report=False):
        """
        Trains the SetFit classifier with another set of data and evaluates its performance.

        Parameters:
        trainset (DataFrame): DataFrame containing training data. Labels are provided in the shape of vectors.
        testset (DataFrame): DataFrame containing test data. Labels are provided in the shape of vectors.
        mcm (bool): Whether to create a multi-label confusion matrix heatmap.
        classification_report (bool): Whether to display a classification report.
        """
        self.categorizer.model = self.categorizer.initialize_model_body()
        metrics = self.categorizer.train_evaluate(trainset, testset, mcm, classification_report)
        print(metrics)