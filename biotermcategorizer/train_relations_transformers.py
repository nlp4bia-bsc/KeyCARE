import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
import sys

# Arguments:
# 1 - traindata path: '/gpfs/scratch/bsc14/bsc14515/jup_lab/data/traindata.tsv'
# 2 - evaldata path: '/gpfs/scratch/bsc14/bsc14515/jup_lab/data/testdata.tsv'
# 3 - base model path: '/gpfs/scratch/bsc14/bsc14515/jup_lab/models/base/sapbert_15_noparents_1epoch.tsv'
# 4 - batch size path: 32
# 5 - output_dir path: '/gpfs/scratch/bsc14/bsc14515/jup_lab/models/trained/NOMBRE_MODELO'

traindata = pd.read_csv(sys.argv[1], sep='\t')
testdata = pd.read_csv(sys.argv[2], sep='\t')

trainX1 = traindata["source"].values.tolist()
trainX2 = traindata["target"].values.tolist()
trainY = traindata["rel_type"].values.tolist()
testX1 = testdata["source"].values.tolist()
testX2 = testdata["target"].values.tolist()
testY = testdata["rel_type"].values.tolist()

tokenizer = AutoTokenizer.from_pretrained(sys.argv[3])
model = AutoModelForSequenceClassification.from_pretrained(sys.argv[3], num_labels=4, problem_type="multi_label_classification")

tokenized_data = tokenizer(trainX1, trainX2, truncation=True, padding=True, return_tensors="pt", max_length=512)
label_strings = [[i] for i in trainY]
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(label_strings)
labels = torch.tensor(labels, dtype=torch.float32)
train_dataset = TensorDataset(tokenized_data.input_ids, tokenized_data.attention_mask, labels)

tokenized_data = tokenizer(testX1, testX2, truncation=True, padding=True, return_tensors="pt", max_length=512)
label_strings = [[i] for i in testY]
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(label_strings)
labels = torch.tensor(labels, dtype=torch.float32)
test_dataset = TensorDataset(tokenized_data.input_ids, tokenized_data.attention_mask, labels)

training_args = TrainingArguments(
    output_dir="./output",  # Output directory
    num_train_epochs=3,     # Number of training epochs
    per_device_train_batch_size=sys.argv[4],  # Batch size per device
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
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
model.save_pretrained(sys.argv[5])