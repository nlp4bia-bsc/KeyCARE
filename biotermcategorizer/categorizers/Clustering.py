from nltk.metrics import distance
import scipy.spatial as spatial
import numpy as np
from scipy.cluster.vq import kmeans
from .Categorizer import Categorizer

class Clustering(Categorizer):
    def __init__(self):
        """
        Initializes the Clustering class.
        """
        self.num_labels = 16

    def compute_clusters(self, keywords):
        """
        Computes clusters for a list of keywords.

        Parameters:
        keywords (list of Keyword): List of Keyword objects to cluster.

        Returns:
        dict: A dictionary where keywords are mapped to their respective clusters.
        """
        word_with_cluster = {}
        words = [kw.text for kw in keywords]
        word_vectors = np.array([
            [
                distance.edit_distance(w, _w)
                for _w in words
            ]
            for w in words
        ], dtype=float)
        
        centroids, _ = kmeans(word_vectors, k_or_guess=min(len(words), self.num_labels))
        
        word_clusters = np.argmin([
            [spatial.distance.euclidean(wv, cv) for cv in centroids]
            for wv in word_vectors
        ], 1)
        
        for i in range(len(words)):
            word_with_cluster[words[i]] = word_clusters[i]
        return word_with_cluster