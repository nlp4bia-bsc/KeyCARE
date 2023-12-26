import os
import re
from nltk.tokenize import word_tokenize
from extractors.RakeExtractor import RakeExtractor
from extractors.YakeExtractor import YakeExtractor
from extractors.TextRankExtractor import TextRankExtractor
from extractors.KeyBertExtractor import KeyBertExtractor
from utils.data_structures import Keyword
from categorizers.Clustering import Clustering
from categorizers.SetFitClassifier import SetFitClassifier
from categorizers.TransformersClassifier import TransformersClassifier

class TermExtractor:
    def __init__(self,
                 extraction_methods=["textrank"], 
                 categorization_method="setfit", 
                 language="spanish", 
                 max_tokens=3, 
                 pos=False,
                 pos_pattern="<NOUN.*>*<ADP.*>*<NOUN.*>*<ADJ.*>*|<PROPN.*>+|<VERB.*>+", #CANVIAR PER EL BO DE VERITAT
                 join=False, 
                 postprocess=True,
                 n=1, 
                 thr_setfit=0.5,
                 thr_transformers=-1,
                 n_clusters=None,
                 categorizer_model_path=None,
                 output_path="./trained_model", 
                 clustering_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                 classifier_model="/mnt/c/Users/Sergi/Desktop/BSC/spanish_sapbert_models/sapbert_15_noparents_1epoch",
                 **kwargs,
                ):
        """
        Initializes the TextCategorizationPipeline.

        Parameters:
        extraction_methods (list): List of keyword extraction methods.
        categorization_method (str): Method for text categorization.
        language (str): Language for text processing.
        max_tokens (int): Maximum number of tokens for keyword extraction.
        pos (bool): Whether to use Part of Speech sequences in KeyBert.
        pos_pattern (str): The Part of Speech regex pattern.
        join (bool): Whether to join keywords from different methods and remove overlaps among them.
        postprocess (bool): Whether to apply postprocessing to keywords.
        n (int): Maximum number of labels for a single sample.
        thr_setfit (float): Threshold for SetFit classification.
        thr_transformers (float): Threshold for Transformers classification.
        n_clusters (int): Number of clusters for clustering.
        categorizer_model_path (str): Path to the categorizer model.
        output_path (str): Path to save the trained model.
        clustering_model (str): Clustering model for keyword clustering.
        classifier_model (str): Classifier model for classification.
        **kwargs: Additional keyword arguments.
        """
        self.extraction_methods = extraction_methods
        self.extractors = self.initialize_keyword_extractors(language, max_tokens, pos, pos_pattern)
        self.categorization_method = categorization_method
        self.categorizer = self.initialize_categorizers(n, thr_setfit, thr_transformers, n_clusters, categorizer_model_path, output_path, clustering_model, classifier_model)
        self.join = join
        self.postprocess = postprocess
        self.kwargs = kwargs
        self.keywords = None
    
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

    def initialize_keyword_extractors(self, language, max_tokens, pos, pos_pattern):
        """
        Initializes keyword extractors based on selected extraction methods.

        Parameters:
        language (str): Language for text processing.
        max_tokens (int): Maximum number of tokens for extracted terms.
        pos (bool): Wether to use Part of Speech sequences in KeyBert.
        pos_pattern (str): The Part of Speech regex pattern.

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

        if 'keybert' in self.extraction_methods:
            keyword_extractors["keybert"] = KeyBertExtractor(language, max_tokens, pos, pos_pattern)

        if not keyword_extractors:
            raise ValueError("No extraction method called {}".format(self.extraction_methods))
        
        return keyword_extractors

    def extract_terms(self, text, join, postprocess):
        """
        Extracts terms from the given text using initialized extractors.

        Parameters:
        text (str): Input text for term extraction.
        join (bool): Whether to join terms obtained with different methods and to remove overlaps among them.
        postprocess (bool): Whether to postprocess the extracted keywords to enhance them.

        Returns:
        list: List of extracted terms.
        """
        try:
            all_terms = []
            for key, extractor in self.extractors.items():
                terms = extractor.extract_terms_without_overlaps(text, kwargs=self.kwargs)
                terms = [term + (key,) for term in terms]
                all_terms.extend(terms)
            if join:
                all_terms = list(self.extractors.values())[0].rmv_overlaps(all_terms)
            if postprocess:
                all_terms = list(self.extractors.values())[0].postprocess_terms(all_terms)
                all_terms = list(self.extractors.values())[0].rmv_overlaps(all_terms)
            self.keywords = [Keyword(text=i[0], extraction_method=i[4], ini=i[1], fin=i[2], score=i[3]) for i in all_terms]
        except:
            raise AttributeError("A list of extractors must be provided")

    def initialize_categorizers(self, n, thr_setfit, thr_transformers, n_clusters, model_path, output_path, clustering_model, classifier_model):
        """
        Initializes the categorizer based on the selected method.

        Parameters:
        n (int): Maximum number of labels for a single mention.
        thr_setfit (float): Threshold for label filtering in SetFitClassifier.
        thr_transformers (float): Threshold for label filtering in TransformersClassifier.
        n_clusters (int): Number of clusters for clustering-based categorization.
        model_path (str): Path to the pretrained model.
        output_path (str): Path to save the trained model.
        clustering_model (str): Clustering model for keyword clustering.
        classifier_model (str): Classifier model for keyword classification.

        Returns:
        Categorizer: Initialized categorizer object.
        """
        if 'transformers' == self.categorization_method:
            categorizer = TransformersClassifier(n, thr_transformers, model_path, output_path, classifier_model)
            
        elif 'setfit' == self.categorization_method:
            categorizer = SetFitClassifier(n, thr_setfit, model_path, output_path, classifier_model)

        elif 'clustering' == self.categorization_method:
            if n_clusters is None:
                raise TypeError("TermExtractor.__init__() missing 1 required positional argument: 'n_clusters' when selecting the Clustering algorithm")
            categorizer = Clustering(n_clusters, model_path, output_path, clustering_model)
        else:
            raise ValueError("No categorizer method called {}".format(self.categorization_method))
        return categorizer

    def categorize_terms(self):
        """
        Categorizes the extracted terms (keyword objects) using the selected categorizer method.
        """
        try:
            if self.categorization_method == 'clustering':
                list_of_keywords = [kw.text for kw in self.keywords]
                clusters = self.categorizer.predict_clusters(list_of_keywords)
                for kw in self.keywords:
                    kw.label = clusters[kw.text]
                    kw.categorization_method = self.categorization_method
            # else:
            #     for kw in self.keywords:
            #         kw.label = self.categorizer.compute_predictions(kw.text)
            #         kw.categorization_method = self.categorization_method
            else:
                labels = self.categorizer.compute_predictions(self.keywords)
                for i in len(self.keywords):
                    self.keywords[i].label = labels[i]
                    self.keywords[i].categorization_method = self.categorization_method
        except:
            raise AttributeError("A categorizer method must be provided")

    def train_classifier(self, trainX, trainY, testX, testY, mcm=False, classification_report=False, **kwargs):
        """
        Trains and evaluates the text categorization model.

        Parameters:
        trainX (list): List of training data (mentions).
        trainY (list): List of training labels.
        testX (list): List of test data (mentions).
        testY (list): List of test labels.
        mcm (bool): Whether to create a multi-label confusion matrix heatmap.
        classification_report (bool): Whether to display a classification report.
        **kwargs: Additional keyword arguments.

        Prints:
        - String: Metrics to evaluate the performance of the catgorizer.
        """
        self.categorizer.initialize_model_body(trainY)
        metrics = self.categorizer.train_evaluate(trainX, trainY, testX, testY, mcm, classification_report, **kwargs)
        print(metrics)

    def train_clustering(self, trainX, **kwargs):
        """
        Trains the text clustering model using the provided data.

        Parameters:
        trainX (list): List of training data (mentions).
        **kwargs: Additional keyword arguments.
        """
        self.categorizer.train_clusters(trainX, **kwargs)