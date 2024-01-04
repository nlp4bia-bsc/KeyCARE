from setfit import SetFitModel, SetFitTrainer
import torch
import pandas as pd
from datasets import Dataset
from .Categorizer import Categorizer
from sklearn.preprocessing import MultiLabelBinarizer

class SetFitClassifier(Categorizer):
    def __init__(self, n, threshold, model_path, output_path, classifier_model):
        """
        Initializes the SetFitClassifier class.

        Parameters:
        n (int): Maximum number of labels for a single mention.
        threshold (float): Threshold for label prediction.
        model_path (str): Path to the pretrained model (can be None).
        output_path (str): Path to save the trained model.
        classifier_model (str): Name of the classifier model.
        """
        super().__init__(n, threshold, model_path, output_path, classifier_model)

    def initialize_pretrained_model(self, model_path):
        """
        Initializes and returns a pretrained and finetuned model (default or specified).

        Parameters:
        model_path (str): Path to the pretrained model (can be None).

        Returns:
        SetFitModel: The pretrained model.
        """
        if model_path is None:
            path = '/mnt/c/Users/Sergi/Desktop/BSC/modelos_entrenados/SetFit/noparents_sp'
            model = SetFitModel.from_pretrained(path)
        else:
            model = SetFitModel.from_pretrained(model_path)
        return model

    def compute_predictions(self, mention):
        """
        Computes label predictions for a list of given mentions.

        Parameters:
        mention (list): List of input mentions for prediction.

        Returns:
        list: List of lists of filtered labels considering the given threshold and the maximum labels.
        """
        final_labels = list()
        mention_text = [m.text for m in mention]
        embeddings = self.model.model_body.encode(mention_text, normalize_embeddings=self.model.normalize_embeddings, convert_to_tensor=True)
        predicts = self.model.model_head.predict_proba(embeddings)
        for j in range(len(predicts[0])):
            predscores = {self.labels[i]: arr[:,1].tolist()[j] for i, arr in enumerate(predicts)}
            top_n_labels = sorted(predscores, key=predscores.get, reverse=True)[:self.n]
            filtered_labels = [label for label in top_n_labels if predscores[label] > self.threshold]
            final_labels.append(filtered_labels)
        return final_labels

    def initialize_model_body(self, trainY):
        """
        Initializes the model body from the given base model.

        Parameters: trainY
        """
        self.model = SetFitModel.from_pretrained(self.classifier_model, multi_target_strategy="multi-output")

    def train_evaluate(self, trainX, trainY, testX, testY, mcm, classification_report, **kwargs):
        """
        Trains and evaluates the SetFit model on the provided data.

        Parameters:
        trainX (list): List of training data (mentions).
        trainY (list): List of training labels.
        testX (list): List of test data (mentions).
        testY (list): List of test labels.
        mcm (bool): Whether to create a multi-label confusion matrix heatmap.
        classification_report (bool): Whether to display a classification report.
        **kwargs: Additional arguments for training.

        Prints:
        - A warning message for specified parameters that cannot be changed when that is attempted.
        """
        train_dataset, test_dataset = self.prepare_data(trainX, trainY, testX, testY)
        evaluate_with_params = self.lambda_evaluate_model(mcm, classification_report)
        specified_params = ['metric', 'num_iterations']
        for param in specified_params:
            if param in kwargs:
                del kwargs[param]
                print("Warning. The parameter " + param + " cannot be changed from its given value.")
        trainer = SetFitTrainer(model=self.model, 
            train_dataset=train_dataset, 
            eval_dataset=test_dataset, 
            metric=evaluate_with_params, 
            num_iterations=5, 
            **kwargs,
        )
        trainer.train()
        metrics = trainer.evaluate()
        self.model.save_pretrained(self.output_path)
        return metrics

    def prepare_data(self, trainX, trainY, testX, testY):
        """
        Prepares training and test datasets for SetFit model training.

        Parameters:
        trainX (list): List of training data (mentions).
        trainY (list): List of training labels.
        testX (list): List of test data (mentions).
        testY (list): List of test labels.

        Returns:
        Tuple: Tuple containing training and test datasets.
        """
        trainY = [{i} for i in trainY]
        testY = [{i} for i in testY]
        mlb = MultiLabelBinarizer()
        mlb.fit_transform(trainY)
        self.labels = [i for i in mlb.classes_]
        train_dataset = Dataset.from_dict({"text": trainX, "label": mlb.fit_transform(trainY)})
        test_dataset = Dataset.from_dict({"text": testX, "label": mlb.transform(testY)})
        return train_dataset, test_dataset
