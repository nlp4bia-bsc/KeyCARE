import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from .Categorizer import Categorizer

class TransformersClassifier(Categorizer):
    def __init__(self, n, threshold, model_path, output_path, classifier_model):
        """
        Initializes the TransformersClassifier class.

        Parameters:
        n (int): Maximum number of labels for a single mention.
        threshold (float): Threshold for label prediction.
        model_path (str): Path to the pretrained model (can be None).
        output_path (str): Path to save the trained model.
        classifier_model (str): Name of the classifier model.
        """
        super().__init__(n, threshold, model_path, output_path, classifier_model)
        
    def initialize_pretrained_model(self):
        """
        Initializes and returns a pretrained and fine-tuned model.

        Returns:
        AutoModelForSequenceClassification: The pretrained model.
        """
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.labels])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.classifier_model is None:
            path = 'BSC-NLP4BIA/biomedical-term-classifier'        
            model = AutoModelForSequenceClassification.from_pretrained(path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.classifier_model)
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
        tokenized_mention = self.tokenizer(mention_text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = self.model(**tokenized_mention)
        logits = output.logits
        for i in range(len(logits.tolist())):
            predscores = {label: score for label, score in zip(self.labels, logits.tolist()[i])}
            top_n_labels = sorted(predscores, key=predscores.get, reverse=True)[:self.n]
            filtered_labels = [label for label in top_n_labels if predscores[label] > self.threshold]
            final_labels.append(filtered_labels)
        return final_labels
    
    def initialize_model_body(self, trainY):
        """
        Initializes the model body for training. That includes finding labels, initializing the tokenizer, and the model for multi-label classification

        Parameters:
        trainY (list): List of training labels.
        """
        self.find_labels(trainY)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=self.num_labels, problem_type="multi_label_classification")

    def train_evaluate(self, trainX, trainY, testX, testY, mcm, classification_report, **kwargs):
        """
        Trains and evaluates the Transformers model on the provided data.

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
        train_dataset = self.prepare_data(trainX, trainY)
        self.train(train_dataset, **kwargs)
        test_dataset = self.prepare_data(testX, testY)
        self.evaluate(test_dataset, mcm, classification_report)
        self.model.save_pretrained(self.output_path)

    def find_labels(self, trainY):
        """
        Finds and sets the unique labels in the training data.

        Parameters:
        trainY (list): List of training labels.
        """
        self.mlb = MultiLabelBinarizer()
        Y = [[i] for i in trainY]
        self.mlb.fit(Y)
        self.labels = [i for i in self.mlb.classes_]
        self.num_labels = len(self.labels)
    
    def prepare_data(self, X, Y):
        """
        Prepares training and test datasets for Transformers model training.

        Parameters:
        X (list): List of data (mentions).
        Y (list): List of labels.
        
        Returns:
        TensorDataset: TensorDataset containing tokenized data and labels.
        """
        tokenized_data = self.tokenizer(X, truncation=True, padding=True, return_tensors="pt", max_length=512)
        label_strings = [[i] for i in Y]
        labels = self.mlb.transform(label_strings)
        labels = torch.tensor(labels, dtype=torch.float32)
        tensordataset = TensorDataset(tokenized_data.input_ids, tokenized_data.attention_mask, labels)
        return tensordataset

    def train(self, train_dataset, **kwargs):
        """
        Trains the Transformers model on the provided dataset.

        Parameters:
        train_dataset (TensorDataset): Dataset for training.
        **kwargs: Additional arguments for training.

        Prints:
        - A warning message for specified parameters that cannot be changed.
        """
        training_args = TrainingArguments(
            output_dir="./output",  
            num_train_epochs=3,  
            per_device_train_batch_size=32, 
            evaluation_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
        )
        specified_params = ['data_collator', 'args','model']
        for param in specified_params:
            if param in kwargs:
                del kwargs[param]
                print("Warning. The parameter " + param + " cannot be changed from its given value.")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.collate_fn, 
            train_dataset=train_dataset,
            **kwargs,
        )
        self.trainer.train()

    def evaluate(self, testset, mcm, classification_report):
        """
        Evaluates the Transformers model on the provided test dataset.

        Parameters:
        testset (TensorDataset): Test dataset.
        mcm (bool): Whether to create a multi-label confusion matrix heatmap.
        classification_report (bool): Whether to display a classification report.

        Prints:
        - Evaluation metrics based on provided options.
        """
        results = self.trainer.predict(testset)
        max_indices = np.argmax(results.predictions, axis=1)
        preds = np.zeros_like(results.predictions)
        preds[np.arange(len(max_indices)), max_indices] = 1
        metrics = self.evaluate_model(preds, results.label_ids, mcm, classification_report)
        print(metrics)
        
    def collate_fn(self, batch):
        """
        Collates batches for training.

        Parameters:
        batch (list): Batch data containing input_ids, attention_mask, and labels.

        Returns:
        dict: Dictionary containing the batched input_ids, attention_mask, and labels.
        """
        return {
        'input_ids': torch.stack([item[0] for item in batch]),
        'attention_mask': torch.stack([item[1] for item in batch]),
        'labels': torch.stack([item[2] for item in batch])
        }
