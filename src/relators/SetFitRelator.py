from setfit import SetFitModel
from .Relator import Relator

class SetFitRelator(Relator):
    def __init__(self, n, threshold, model_path):
        """
        Initializes SetFitRelator, a class inheriting from Relator.

        Parameters:
        n (int): Maximum number of labels for a single relation.
        threshold (float): Threshold value for mentions relation.
        model_path (str): Path to the model.
        """
        super().__init__(n, threshold, model_path)
    
    def initialize_pretrained_model(self, model_path):
        """
        Initializes a pretrained SetFitModel based on the provided model_path.

        Parameters:
        model_path (str): Path to the pretrained model.

        Returns:
        object: SetFitModel instance.
        """
        if model_path is None:
            path = 'BSC-NLP4BIA/biomedical-semantic-relation-classifier-setfit'
            model = SetFitModel.from_pretrained(path)
        else:
            model = SetFitModel.from_pretrained(model_path)
        return model

    def compute_relation(self, source, target):
        """
        Computes relations between source and target entities.

        Parameters:
        source (list): List of source entities.
        target (list): List of target entities.

        Returns:
        list: List of labels representing computed relations.
        """
        final_labels = list()
        mentions = list()
        for i in range(len(source)):
            mentions.append(source[i].text + " </s> " + target[i].text)
        embeddings = self.model.model_body.encode(mentions, normalize_embeddings=self.model.normalize_embeddings, convert_to_tensor=True)
        predicts = self.model.model_head.predict_proba(embeddings)
        for j in range(len(predicts[0])):
            predscores = {self.labels[i]: arr[:,1].tolist()[j] for i, arr in enumerate(predicts)}
            top_n_labels = sorted(predscores, key=predscores.get, reverse=True)[:self.n]
            filtered_labels = [label for label in top_n_labels if predscores[label] > self.threshold]
            final_labels.append(filtered_labels)
        return final_labels