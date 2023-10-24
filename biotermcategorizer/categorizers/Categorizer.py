from sklearn.preprocessing import MultiLabelBinarizer
import evaluate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

class Categorizer:
    def __init__(self, n, threshold, model_path, output_path, classifier_model):
        """
        Initializes the Categorizer class.

        Parameters:
        n (int): Number of maximum labels for a single mention.
        threshold (float): Threshold for classification of a sample.
        model_path (str): Path to the pretrained model.
        output_path (str): Path to save the output.
        classifier_model (str): Name of the classifier model.
        """
        self.n = n
        self.threshold = threshold
        self.labels = ['ACTIVIDAD', 'COMUNIDAD', 'DEPARTAMENTO', 'ENFERMEDAD', 'FAC_GEN', 'FAC_NOM', 'FARMACO', 'GEO_GEN', 'GEO_NOM', 'GPE_GEN', 'GPE_NOM', 'HUMAN', 'IDIOMA', 'MORFOLOGIA_NEOPLASIA', 'NO_CATEGORY', 'PROCEDIMIENTO', 'PROFESION', 'SINTOMA', 'SITUACION_LABORAL', 'SPECIES', 'TRANSPORTE']
        self.output_path = output_path
        self.classifier_model = classifier_model
        self.model = self.initialize_pretrained_model(model_path)

    def evaluate_model(self, y_pred, y_test, mcm, classification_report):
        """
        Evaluates the model's performance and calls the selected evauation functions.

        Parameters:
        y_pred (list): Predicted labels. Labels are provided in the shape of vectors.
        y_test (list): True labels. Labels are provided in the shape of vectors.
        mcm (bool): Whether to create a multi-label confusion matrix heatmap.
        classification_report (bool): Whether to display a classification report.

        Returns:
        dict: Dictionary containing computed evaluation metrics.
        """
        if classification_report:
            self.classification_report(y_pred, y_test)
        if mcm:
            self.mcm_heatmap(y_pred, y_test)
        return self.compute_metrics(y_pred, y_test)

    def lambda_evaluate_model(self, mcm, classification_report):
        """
        Returns a lambda function for evaluating the model.

        Parameters:
        mcm (bool): Whether to create a multi-label confusion matrix heatmap.
        classification_report (bool): Whether to display a classification report.

        Returns:
        function: Lambda function for model evaluation.
        """
        return lambda y_pred, y_test: self.evaluate_model(y_pred, y_test, mcm, classification_report)

    def compute_metrics(self, y_pred, y_test):
        """
        Computes evaluation metrics: F-score, accuracy, and number of samples within each class with no given label.

        Parameters:
        y_pred (list): Predicted labels. Labels are provided in the shape of vectors.
        y_test (list): True labels. Labels are provided in the shape of vectors.

        Returns:
        dict: Dictionary containing computed evaluation metrics.
        """
        multilabel_f1_metric = evaluate.load("f1", "multilabel")
        multilabel_accuracy_metric = evaluate.load("accuracy", "multilabel")
        f1 = multilabel_f1_metric.compute(predictions=y_pred, references=y_test, average="micro")["f1"]
        accuracy = multilabel_accuracy_metric.compute(predictions=y_pred, references=y_test)["accuracy"]

        y_pred = np.array(y_pred)
        y_test = np.array(y_test)

        no_label_samples = []
        for idx, pred in enumerate(y_pred):
            if np.all(pred == 0):
                true_labels = [self.labels[i] for i, value in enumerate(y_test[idx]) if value == 1]
                no_label_samples.extend(true_labels)

        label_counts = Counter(no_label_samples)
        label_counts_dict = dict(label_counts)
        return {"f1": f1, "accuracy": accuracy, "Classes with no given label": label_counts_dict}

    def classification_report(self, y_pred, y_test):
        """
        Generates and displays a classification report.

        Parameters:
        y_pred (list): Predicted labels. Labels are provided in the shape of vectors.
        y_test (list): True labels. Labels are provided in the shape of vectors.
        """
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)

        print(classification_report(y_test, y_pred, target_names=self.labels))

    def mcm_heatmap(self, y_pred, y_test):
        """
        Generates and displays a multi-label confusion matrix heatmap.

        Parameters:
        y_pred (list): Predicted labels. Labels are provided in the shape of vectors.
        y_test (list): True labels. Labels are provided in the shape of vectors.
        """
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)

        samples_with_predictions = np.any(y_pred, axis=1)

        y_pred_with_predictions = y_pred[samples_with_predictions]
        y_test_with_predictions = y_test[samples_with_predictions]

        confusion_matrix_multi = confusion_matrix(y_test_with_predictions.argmax(axis=1), y_pred_with_predictions.argmax(axis=1), labels=np.arange(len(self.labels)))
    
        plt.figure(figsize=(10, 7))
        plt.imshow(confusion_matrix_multi, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Multi-Label Confusion Matrix")
        plt.colorbar()

        tick_marks = np.arange(len(self.labels))
        plt.xticks(tick_marks, self.labels, rotation=45)
        plt.yticks(tick_marks, self.labels)

        for i in range(len(self.labels)):
            for j in range(len(self.labels)):
                plt.text(j, i, str(confusion_matrix_multi[i, j]), ha="center", va="center", color="white" if confusion_matrix_multi[i, j] > confusion_matrix_multi.max() / 2 else "black")

        plt.tight_layout()
        plt.ylabel("True Labels")
        plt.xlabel("Predicted Labels")
        plt.show()
