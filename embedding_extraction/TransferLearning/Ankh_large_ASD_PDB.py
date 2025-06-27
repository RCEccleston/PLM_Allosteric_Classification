
# -*- coding: utf-8 -*-

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import ast

import torch
import numpy as np
import random

seed = 7

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import ankh

from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import Trainer, TrainingArguments, EvalPrediction
from datasets import load_dataset

from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats
from functools import partial
import pandas as pd
from tqdm.auto import tqdm
def get_num_params(model):
    return sum(p.numel() for p in model.parameters())

import evaluate
from sklearn import metrics
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
from transformers import TrainingArguments, Trainer, set_seed

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Available device:', device)

model, tokenizer = ankh.load_large_model()
model.eval()
model.to(device=device)
print(f"Number of parameters:", get_num_params(model))

# Set random seeds for reproducibility of your trainings run
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)


# Set all random seeds
seed= 42      #random seed
set_seeds(seed)

def make_mask(sequences):
  masks = []
  for seq in sequences:
    masks.append([1]*len(seq))
  return masks

train_file = '~/PLM_Allosteric_Classification/data/train_df.csv'
test_file = '~/PLM_Allosteric_Classification/data/test_df.csv'
train_df = pd.read_csv(train_file)
test_df_total = pd.read_csv(test_file)
train_sequences = train_df['Sequences'].tolist()
train_labels = train_df['Labels'].apply(ast.literal_eval).tolist()
N_test_total = len(test_df_total)

# Generate a random permutation of indices

indices = np.random.permutation(N_test_total)

# Split the indices in two, e.g., 70% and 30%

split_index = int(0.5 * N_test_total)
valid_indices = indices[:split_index]
test_indices = indices[split_index:]

test_df = test_df_total.iloc[test_indices]
valid_df = test_df_total.iloc[valid_indices]

test_sequences = test_df['Sequences'].tolist()
test_labels = test_df['Labels'].apply(ast.literal_eval).tolist()

validation_sequences = valid_df['Sequences'].tolist()
validation_labels = valid_df['Labels'].apply(ast.literal_eval).tolist()

print(len(train_sequences))
print(len(validation_sequences))
print(len(test_sequences))

assert len(train_sequences) == len(train_labels), "Train sequences and labels length mismatch"
assert len(validation_sequences) == len(validation_labels), "Validation sequences and labels length mismatch"
assert len(test_sequences) == len(test_labels), "Test sequences and labels length mismatch"


my_train = {"sequence" : train_sequences, "label" : train_labels}
my_train = pd.DataFrame(my_train)
my_valid = {"sequence" : validation_sequences, "label" : validation_labels}
my_valid = pd.DataFrame(my_valid)
my_test = {"sequence" : test_sequences, "label" : test_labels}
my_test = pd.DataFrame(my_test)


allo_training_sequences = my_train["sequence"]
allo_training_labels = my_train["label"]

allo_valid_sequences = my_valid["sequence"]
allo_valid_labels = my_valid["label"]

allo_test_sequences = my_test["sequence"]
allo_test_labels = my_test["label"]

print(allo_training_sequences)

def preprocess_dataset(sequences, labels, max_length=None):
    sequences = sequences.astype(str)
    sequences = ["".join(seq.split()) for seq in sequences]

    if max_length is None:
        max_length = len(max(sequences, key=lambda x: len(x)))

    seqs = [list(seq)[:max_length] for seq in sequences]
    labels = [list(label)[:max_length] for label in labels]
    labels = [[str(num) for num in label] for label in labels]


    assert len(seqs) == len(labels)
    return seqs, labels

def embed_dataset(model, sequences, shift_left = 0, shift_right = -1):
    inputs_embedding = []
    with torch.no_grad():
        for sample in tqdm(sequences):
            l = len(sample)
            print(l)
            ids = tokenizer.batch_encode_plus([sample], add_special_tokens=True,
                                              padding=True, is_split_into_words=True,
                                              return_tensors="pt")
            embedding = model(input_ids=ids['input_ids'].to(device))[0]
            embedding = embedding[0].detach().cpu().numpy()[shift_left:shift_right]
#            print(embedding.shape())
            inputs_embedding.append(embedding[:l,:])
    return inputs_embedding

training_sequences, training_labels = preprocess_dataset(allo_training_sequences, allo_training_labels)
validation_sequences, validation_labels = preprocess_dataset(allo_valid_sequences, allo_valid_labels)
test_sequences, test_labels = preprocess_dataset(allo_test_sequences, allo_test_labels)
[print(len(seq)) for seq in training_sequences]
training_embeddings = embed_dataset(model, training_sequences)
validation_embeddings = embed_dataset(model, validation_sequences)
test_embeddings = embed_dataset(model, test_sequences)

id2tag = {0: "0", 1: "1"}
tag2id = {"0": 0, "1":1}
def encode_tags(labels):
    labels = [[tag2id[tag] for tag in doc] for doc in labels]
    return labels


train_labels_encodings = encode_tags(training_labels)
validation_labels_encodings = encode_tags(validation_labels)
test_labels_encodings = encode_tags(test_labels)


class AlloDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        embedding = self.encodings[idx]
        labels = self.labels[idx]
        return {'embed': torch.tensor(embedding), 'labels': torch.tensor(labels)}

    def __len__(self):
        return len(self.labels)

training_dataset = AlloDataset(training_embeddings, train_labels_encodings)
validation_dataset = AlloDataset(validation_embeddings, validation_labels_encodings)
test_dataset = AlloDataset(test_embeddings, test_labels_encodings)

def compute_weights(training_labels):
  total_samples=0
  class_0_samples=0
  class_1_samples=0
  for seq in training_labels:
    total_samples+=len(seq)
    class_0_samples+=seq.count('0') ## this is for if the labels are strings
    class_1_samples+=seq.count('1')
  weight_0 = total_samples / (2 * class_0_samples)
  weight_1 = total_samples / (2 * class_1_samples)

  return weight_0, weight_1

weight_0, weight_1 = compute_weights(training_labels)

def compute_frac(validation_labels):
  total_samples=0
  class_0_samples=0
  class_1_samples=0
  for seq in training_labels:
    total_samples+=len(seq)
    class_0_samples+=seq.count('0') ## this is for if the labels are strings
    class_1_samples+=seq.count('1')
  weight_0 = total_samples / (2 * class_0_samples)
  weight_1 = total_samples / (2 * class_1_samples)

  frac_0 = class_0_samples/total_samples
  frac_1 = class_1_samples/total_samples
  return frac_0, frac_1

frac_0, frac_1 = compute_frac(validation_labels)
print(frac_0, frac_1)

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray):
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(id2tag[label_ids[i][j]])
                    preds_list[i].append(id2tag[preds[i][j]])

        return preds_list, out_label_list


def compute_metrics(p):
    logits = p.predictions.reshape(-1, p.predictions.shape[-1])
    labels = p.label_ids.reshape(-1)
    probs = softmax(logits, axis=1)
    probs = probs[:, -1].reshape(-1)

    # Filter out invalid labels
    probs = probs[labels != -100]
    labels = labels[labels != -100]

    # Calculate AUC and APS
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc_value = auc(fpr, tpr)
    aps = average_precision_score(labels, probs)

    # Binarize the probabilities for precision, recall, F1 score, and accuracy
    threshold = 0.5
    preds = (probs >= threshold).astype(int)

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    accuracy = accuracy_score(labels, preds)

    return {
        "AUC": auc_value,
        "APS": aps,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy
    }


def model_init(num_tokens, embed_dim):
    hidden_dim = int(embed_dim / 2)
    num_hidden_layers = 1 # Number of hidden layers in ConvBert.
    nlayers = 1 # Number of ConvBert layers.
    nhead = 4
    dropout = 0.2
    conv_kernel_size = 7
    downstream_model = ankh.ConvBertForMultiClassClassification(num_tokens=num_tokens,
                                                                input_dim=embed_dim,
                                                                nhead=nhead,
                                                                hidden_dim=hidden_dim,
                                                                num_hidden_layers=num_hidden_layers,
                                                                num_layers=nlayers,
                                                                kernel_size=conv_kernel_size,
                                                                dropout=dropout)
    return downstream_model.cuda()
model_embed_dim = 1536
model_type = 'ankh_large'
experiment = f'ASD_PDB_{model_type}'

model_path  = '~/PLM_Allosteric_Classification/embedding_extraction/TransferLearning/Ankh_PDB.ckpt'

pdb_model = model_init(num_tokens=2, embed_dim=model_embed_dim)
pdb_model.load_state_dict(torch.load(model_path))


training_args = TrainingArguments(
    output_dir=f'~/PLM_Allosteric_Classification/embedding_extraction/TransferLearning/results_{experiment}',
    num_train_epochs=40,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=1000,
    learning_rate=1e-04,
    weight_decay=0.001,
    logging_dir=f'./logs_{experiment}',
    logging_steps=200,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=16,
    fp16=False,
    fp16_opt_level="02",
    run_name=experiment,
    seed=seed,
    load_best_model_at_end=True,
    metric_for_best_model="eval_APS",
    greater_is_better=True,
    save_strategy="epoch"
)

class CustomTrainer(Trainer):
    def __init__(self, weight_0, weight_1, **kwargs):
        super().__init__(**kwargs)
        self.weights=torch.tensor([weight_0, weight_1]).float().cuda()
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(inputs.get("embed"))
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=self.weights)
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

model_embed_dim = 1536
trainer = CustomTrainer(
    weight_0=weight_0,
    weight_1=weight_1,
    model=pdb_model,
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

predictions, labels, metrics_output = trainer.predict(test_dataset)
metrics_output

print(trainer.state.log_history)
pd.set_option('display.width', None)

print("test metrics")
print(metrics_output)

path = '~/PLM_Allosteric_Classification/TransferLearning/Ankh_large_ASD_PDB.ckpt'
torch.save(trainer.model.state_dict(), path)


