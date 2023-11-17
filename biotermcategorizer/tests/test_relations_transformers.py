import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
from collections import Counter

path='/gpfs/scratch/bsc14/bsc14515/jup_lab/data/traindata_relations.tsv'
traindata = pd.read_csv(path, sep='\t')
path='/gpfs/scratch/bsc14/bsc14515/jup_lab/data/testdata_relations.tsv'
testdata = pd.read_csv(path, sep='\t')
traindata["source_target"]=traindata["source"] + " </s> " + traindata["target"]
testdata["source_target"]=testdata["source"] + " </s> " + testdata["target"]

path='/gpfs/scratch/bsc14/bsc14515/jup_lab/data/traindata_relations_nocategory.tsv'
traindata_nocategory = pd.read_csv(path, sep='\t')
path='/gpfs/scratch/bsc14/bsc14515/jup_lab/data/testdata_relations_nocategory.tsv'
testdata_nocategory = pd.read_csv(path, sep='\t')
traindata_nocategory["source_target"]=traindata_nocategory["source"] + " </s> " + traindata_nocategory["target"]
testdata_nocategory["source_target"]=testdata_nocategory["source"] + " </s> " + testdata_nocategory["target"]

traindata_narrow = traindata[traindata["rel_type"]=="NARROW"].head(25000)
traindata_broad = traindata[traindata["rel_type"]=="BROAD"].head(25000)
traindata_exact = traindata[traindata["rel_type"]=="EXACT"].head(12500)
traindata_nocat = traindata_nocategory.head(37500)
traindata_head = pd.concat([traindata_narrow,traindata_broad,traindata_exact,traindata_nocat], axis=0)
testdata_head = pd.concat([testdata.head(22500),testdata_nocategory.head(10000)], axis=0)

trainX1 = traindata_head["source"].values.tolist()
trainX2 = traindata_head["target"].values.tolist()
trainY = traindata_head["rel_type"].values.tolist()
testX1 = testdata_head["source"].values.tolist()
testX2 = testdata_head["target"].values.tolist()
testY = testdata_head["rel_type"].values.tolist()

model_path = '/gpfs/scratch/bsc14/bsc14515/jup_lab/models/base/sapbert_15_parents_1epoch'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4, problem_type="multi_label_classification")

tokenized_data = tokenizer(trainX1, trainX2, truncation=True, padding=True, return_tensors="pt", max_length=512)
label_strings = [[i] for i in trainY]
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(label_strings)
labels = torch.tensor(labels, dtype=torch.float32)
dataset = TensorDataset(tokenized_data.input_ids, tokenized_data.attention_mask, labels)

tokenized_data = tokenizer(testX1, testX2, truncation=True, padding=True, return_tensors="pt", max_length=512)
label_strings = [[i] for i in testY]
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(label_strings)
labels = torch.tensor(labels, dtype=torch.float32)
dataset2 = TensorDataset(tokenized_data.input_ids, tokenized_data.attention_mask, labels)

training_args = TrainingArguments(
    output_dir="./output",  # Output directory
    num_train_epochs=3,     # Number of training epochs
    per_device_train_batch_size=32,  # Batch size per device
    evaluation_strategy="steps",  # Evaluate every steps
    save_steps=500,  # Save checkpoint every 500 steps
    save_total_limit=2,  # Only keep the last 2 checkpoints
    load_best_model_at_end=True,  # Load the best model at the end of training
)

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item[0] for item in batch]),
        'attention_mask': torch.stack([item[1] for item in batch]),
        'labels': torch.stack([item[2] for item in batch])
    }

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,  # You can customize data collation if needed
    train_dataset=dataset,
    eval_dataset=dataset2,
)

trainer.train()
model.save_pretrained('/gpfs/scratch/bsc14/bsc14515/jup_lab/models/trained/transformers_trained_model')
results = trainer.predict(dataset2)

max_indices = np.argmax(results.predictions, axis=1)
preds = np.zeros_like(results.predictions)
preds[np.arange(len(max_indices)), max_indices] = 1

with open('predictions.txt', 'w') as file:
    file.write(preds)

f1 = f1_score(preds, results.label_ids, average="micro")  # You can choose other average options as needed
accuracy = accuracy_score(preds, results.label_ids)

def mcm_heatmap(y_pred, y_test):
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    
    samples_with_predictions = np.any(y_pred, axis=1)

    # Filter y_pred and y_test to include only samples with predictions
    y_pred_with_predictions = y_pred[samples_with_predictions]
    y_test_with_predictions = y_test[samples_with_predictions]

    # Initialize the confusion matrix
    confusion_matrix_multi = confusion_matrix(y_test_with_predictions.argmax(axis=1), y_pred_with_predictions.argmax(axis=1), labels=np.arange(4))
    return confusion_matrix_multi

with open('evaluation.txt', 'w') as file:
    file.write("F1-score:", f1)
    file.write("Accuracy:", accuracy)
    file.write(classification_report(results.label_ids, preds, target_names=['BROAD','EXACT','NARROW','NO_RELATION']))
    file.write(mcm_heatmap(results.label_ids, preds))