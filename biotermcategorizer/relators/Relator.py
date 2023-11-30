class Relator:
    def __init__(self, n, threshold, model_path):
        """
        Initializes the Relator class.

        Parameters:
        n (int): Maximum number of labels for a single relation.
        threshold (float): Threshold value used in the relation extraction.
        model_path (str): Path to the model.
        """
        self.n = n
        self.threshold = threshold
        self.labels = ['BROAD','EXACT','NARROW','NO_CATEGORY']
        self.model = self.initialize_pretrained_model(model_path)