class Relator:
    def __init__(self, n, threshold, model_path):
        self.n = n
        self.threshold = threshold
        self.labels = ['BROAD','EXACT','NARROW']
        self.model = self.initialize_pretrained_model(model_path)