from setfit import SetFitModel
from .Relator import Relator

class SetFitRelator(Relator):
    def __init__(self, n, threshold, model_path):
        super().__init__(n, threshold, model_path)
    
    def initialize_pretrained_model(self, model_path):
        if model_path is None:
            path = '/mnt/c/Users/Sergi/Desktop/BSC/modelos_entrenados/setfit_rel1'
            model = SetFitModel.from_pretrained(path)
        else:
            model = SetFitModel.from_pretrained(model_path)
        return model

    def compute_relation(self, source, target):
        mention = source + " </s> " + target
        embeddings = self.model.model_body.encode([mention], normalize_embeddings=self.model.normalize_embeddings, convert_to_tensor=True)
        predicts = self.model.model_head.predict_proba(embeddings)
        predscores = {self.labels[i]: arr[:,1].tolist()[0] for i, arr in enumerate(predicts)}
        top_n_labels = sorted(predscores, key=predscores.get, reverse=True)[:self.n]
        filtered_labels = [label for label in top_n_labels if predscores[label] > self.threshold]
        return filtered_labels