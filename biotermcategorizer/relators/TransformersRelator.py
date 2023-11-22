import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .Relator import Relator

class TransformersRelator(Relator):
    def __init__(self, n, threshold, model_path):
        super().__init__(n, threshold, model_path)
        
    def initialize_pretrained_model(self, model_path):
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.labels])
        path = '/mnt/c/Users/Sergi/Desktop/BSC/spanish_sapbert_models/sapbert_15_noparents_1epoch'
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if model_path is None:
            path = '/mnt/c/Users/Sergi/Desktop/BSC/modelos_entrenados/transformers_rel1'        
            model = AutoModelForSequenceClassification.from_pretrained(path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return model

    def compute_relation(self, source, target):
        final_labels = list()
        tokenized_mention = self.tokenizer(source, target, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = self.model(**tokenized_mention)
        logits = output.logits
        for i in range(len(logits.tolist())):
            predscores = {label: score for label, score in zip(self.labels, logits.tolist()[i])}
            top_n_labels = sorted(predscores, key=predscores.get, reverse=True)[:self.n]
            filtered_labels = [label for label in top_n_labels if predscores[label] > self.threshold]
            final_labels.append(filtered_labels)
        return final_labels