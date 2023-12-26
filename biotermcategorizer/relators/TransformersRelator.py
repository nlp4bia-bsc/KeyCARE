import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .Relator import Relator

class TransformersRelator(Relator):
    def __init__(self, n, threshold, model_path):
        """
        Initializes TransformersRelator, a class inheriting from Relator.

        Parameters:
        n (int): Maximum number of labels for a single relation.
        threshold (float): Threshold value used for mentions relation.
        model_path (str): Path to the model class.
        """
        super().__init__(n, threshold, model_path)
        
    def initialize_pretrained_model(self, model_path):
        """
        Initializes a pretrained model for TransformersRelator based on the provided model_path.

        Parameters:
        model_path (str): Path to the pretrained model.

        Returns:
        object: Pretrained model instance.
        """
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.labels])
        path = '/mnt/c/Users/Sergi/Desktop/BSC/spanish_sapbert_models/sapbert_15_noparents_1epoch'
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if model_path is None:
            path = '/mnt/c/Users/Sergi/Desktop/BSC/modelos_entrenados/transformers_rel1_B'        
            model = AutoModelForSequenceClassification.from_pretrained(path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return model

    def compute_relation(self, source, target):
        """
        Computes relations between source and target entities using Transformers model.

        Parameters:
        source (list): List of source entities.
        target (list): List of target entities.

        Returns:
        list: List of labels representing computed relations.
        """
        final_labels = list()
        source_text = [s.text for s in source]
        target_text = [t.text for t in target]
        tokenized_mention = self.tokenizer(source_text, target_text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = self.model(**tokenized_mention)
        logits = output.logits
        for i in range(len(logits.tolist())):
            predscores = {label: score for label, score in zip(self.labels, logits.tolist()[i])}
            top_n_labels = sorted(predscores, key=predscores.get, reverse=True)[:self.n]
            filtered_labels = [label for label in top_n_labels if predscores[label] > self.threshold]
            final_labels.append(filtered_labels)
        return final_labels