from sklearn.cluster import KMeans
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import pickle

class Clustering(Categorizer):
    def __init__(self, n_clusters, model_path, output_path, clustering_model):
        """
        Initializes the Clustering class.

        Parameters:
        n_clusters (int): Number of clusters to create.
        model_path (str): Path to the pretrained clustering model (can be None).
        output_path (str): Path to save the trained clustering model.
        clustering_model (str): Name of the clustering model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(clustering_model)
        self.model = AutoModel.from_pretrained(clustering_model)
        self.trained = False
        self.n_clusters = n_clusters
        self.output_path = output_path
        self.model_path = model_path
        
    def generate_embeddings(self, mentions):
        """
        Generates embeddings for a list of text mentions.

        Parameters:
        mentions (list): List of text mentions to generate embeddings for.

        Returns:
        torch.Tensor: Tensor containing the generated embeddings.
        """
        tokenized_inputs = self.tokenizer(mentions, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
        embeddings = outputs.last_hidden_state
        return embeddings[:, 0, :]

    def train_clusters(self, mentions_train, **kwargs):
        """
        Trains a clustering model using the provided text mentions.

        Parameters:
        mentions_train (list): List of text mentions for training.
        **kwargs: Additional arguments for KMeans clustering.

        Prints:
        - A warning message for specified parameters that cannot be changed when that is attempted.
        - Examples of mentions in each cluster (no more than 5 each).
        """
        mentions_embeddings_train = self.generate_embeddings(mentions_train)
        specified_params = ['random_state', 'n_init']
        for param in specified_params:
            if param in kwargs:
                del kwargs[param]
                print("Warning. The parameter " + param + " cannot be changed from its given value.")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto", **kwargs).fit(mentions_embeddings_train)
        self.trained = True
        self.save_trained_model()

        clusters_examples = dict()
        for i in range(max(self.kmeans.labels_)+1):
            instances = []
            for j in range(len(self.kmeans.labels_)):
                if self.kmeans.labels_[j] == i and len(instances) < 5:
                    instances.append(mentions_train[j])
            clusters_examples["Class " + str(i)] = instances
        print(clusters_examples)

    def predict_clusters(self, mentions):
        """
        Predicts clusters for a list of text mentions.

        Parameters:
        mentions (list): List of text mentions for cluster prediction.

        Returns:
        dict: Dictionary mapping mentions to their predicted clusters.
        """
        word_with_cluster = {}
        mentions_embeddings = self.generate_embeddings(mentions)
        if self.model_path is not None and not self.trained:
            self.import_pretrained_model(self.model_path)
            self.trained = True
        elif not self.trained:
            self.train_clusters(mentions)
            self.trained = True
            self.save_trained_model()
        predicted_clusters = self.kmeans.predict(mentions_embeddings)
        for i in range(len(mentions)):
            word_with_cluster[mentions[i]] = predicted_clusters[i]
        return word_with_cluster
    
    def save_trained_model(self):
        """
        Saves the trained KMeans clustering model to the specified output path.
        """
        with open(self.output_path, 'wb') as file:
            pickle.dump(self.kmeans, file)      

    def import_pretrained_model(self, model_path):
        """
        Imports a pretrained clustering model from a file.

        Parameters:
        model_path (str): Path to the pretrained clustering model file.
        """
        with open(model_path, 'rb') as file:
            self.kmeans = pickle.load(file)
        self.trained = True
