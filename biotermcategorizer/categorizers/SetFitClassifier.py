from setfit import SetFitModel, SetFitTrainer
import torch
import pandas as pd
from datasets import Dataset
from .Categorizer import Categorizer

class SetFitClassifier(Categorizer):
    def __init__(self, parents, n, threshold):
        """
        Initializes the SetFitClassifier.

        Parameters:
        parents (bool): Whether to use the SetFit model with parent or not.
        n (int): The maximum number of predicted labels to consider.
        threshold (float): The threshold for label filtering.

        Returns:
        None
        """
        self.parents = parents
        self.n = n
        self.threshold = threshold
        self.model = self.initialize_pretrained_model()
        self.labels = ['ACTIVIDAD', 'COMUNIDAD', 'DEPARTAMENTO', 'ENFERMEDAD', 'FAC_GEN','FAC_NOM', 'FARMACO', 'GEO_GEN', 'GEO_NOM', 'GPE_GEN', 'GPE_NOM','HUMAN', 'IDIOMA', 'MORFOLOGIA_NEOPLASIA', 'NO_CATEGORY','PROCEDIMIENTO', 'PROFESION', 'SINTOMA', 'SITUACION_LABORAL','SPECIES', 'TRANSPORTE']
        
    def initialize_pretrained_model(self):
        """
        Initializes and returns a pretrained and finetuned model based on the 'parents' flag.

        Parameters:
        None

        Returns:
        SetFitModel: The pretrained model.
        """
        if self.parents:
            path = '/mnt/c/Users/Sergi/Desktop/BSC/modelos_entrenados/parents_sp'
        else:
            path = '/mnt/c/Users/Sergi/Desktop/BSC/modelos_entrenados/noparents_sp'
        model = SetFitModel.from_pretrained(path)
        return model

    def compute_predictions(self, mention):
        """
        Computes label predictions for a given mention.

        Parameters:
        mention (str): The input mention for prediction.

        Returns:
        list: List of filtered labels considering the given threshold and the maximum labels.
        """
        embeddings = self.model.model_body.encode([mention], normalize_embeddings=self.model.normalize_embeddings, convert_to_tensor=True)
        predicts = self.model.model_head.predict_proba(embeddings)
        predscores = {self.labels[i]: arr[:,1].tolist()[0] for i, arr in enumerate(predicts)}
        top_n_labels = sorted(predscores, key=predscores.get, reverse=True)[:self.n]
        filtered_labels = [label for label in top_n_labels if predscores[label] > self.threshold]
        return filtered_labels

    def initialize_model_body(self):
        """
        Initializes the model body without finetuning based on the 'parents' flag.

        Parameters:
        None

        Returns:
        SetFitModel: The initialized model.
        """
        if self.parents:
            path = '/mnt/c/Users/Sergi/Desktop/BSC/spanish_sapbert_models/sapbert_15_parents_1epoch'
        else:
            path = '/mnt/c/Users/Sergi/Desktop/BSC/spanish_sapbert_models/sapbert_15_noparents_1epoch'
        model = SetFitModel.from_pretrained(path, multi_target_strategy="multi-output")
        return model

    def train_evaluate(self, trainset, testset, mcm, classification_report):
        """
        Trains and evaluates the model using provided datasets and metrics.

        Parameters:
        trainset (DataFrame): DataFrame containing training data. Must have a column called "text" for the mention and a column called "label" with the class.
        testset (DataFrame): DataFrame containing test data. Must have a column called "text" for the mention and a column called "label" with the class.
        mcm (bool): Whether to produce the heatmap of the multilabel confusion matrix during evaluation.
        classification_report (bool): Whether to show the classification report during evaluation.

        Returns:
        dict: Dictionary containing evaluation metrics.
        """
        train_dataset, test_dataset = self.prepare_data(trainset, testset)
        evaluate_with_params = self.lambda_evaluate_model(mcm, classification_report)
        trainer = SetFitTrainer(model=self.model, train_dataset=train_dataset, eval_dataset=test_dataset, metric=evaluate_with_params, num_iterations=5)
        trainer.train()
        metrics = trainer.evaluate()
        return metrics

    def prepare_data(self, trainset, testset):
        """
        Prepares training and test datasets for model training and evaluation.

        Parameters:
        trainset (DataFrame): DataFrame containing training data. Must have a column called "text" for the mention and a column called "label" with the class.
        testset (DataFrame): DataFrame containing test data. Must have a column called "text" for the mention and a column called "label" with the class.
        
        Returns:
        tuple: Tuple containing train and test datasets.
        """
        trainY=[]
        testY=[]
        for index, row in trainset.iterrows():
            trainY.append({row["label"]})
        for index, row in testset.iterrows():
            testY.append({row["label"]})
        mlb = MultiLabelBinarizer()
        mlb.fit_transform(trainY)
        self.labels = [i for i in mlb.classes_]
        train_dataset = Dataset.from_dict({"text": trainset['text'], "label": mlb.fit_transform(trainY)})
        test_dataset = Dataset.from_dict({"text": testset['text'], "label": mlb.fit_transform(testY)})
        return train_dataset, test_dataset
