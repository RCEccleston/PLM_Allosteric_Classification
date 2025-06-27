from transformers import Trainer, TrainingArguments, EvalPrediction, EsmForTokenClassification, set_seed
import esm
import torch
torch.cuda.empty_cache()
from torch.cuda.amp import autocast

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset, DataLoader

import re
import numpy as np
import pandas as pd
import copy
import ankh
from functools import partial
import evaluate
from sklearn import metrics
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
import random
import ast
from sklearn.model_selection import train_test_split
#from datasets import Dataset
from tqdm.auto import tqdm
from datasets import Dataset as HFDataset
import evaluate
from sklearn.metrics import average_precision_score
from sklearn import metrics
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
from scipy.special import softmax
from Bio.Align import PairwiseAligner

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Available device:', device)

# Set random seeds for reproducibility of your trainings run
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)


# Set all random seeds
seed= 42      #random seed
set_seeds(seed)

train_file = '~/PLM_Allosteric_Classification/data/train_df_pdb.csv'
test_file = '~/PLM_Allosteric_Classification/data/test_df_pdb.csv'
train_df = pd.read_csv(train_file)
test_df_total = pd.read_csv(test_file)
train_sequences = train_df['Sequences'].tolist()
train_labels = train_df['Labels'].apply(ast.literal_eval).tolist()
N_test_total = len(test_df_total)

N_test_total = len(test_df_total)
print(N_test_total)

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


# Check the types of the split sequences
print(f"Type of train_sequences: {type(train_sequences)}, example: {train_sequences[:2]}")
print(f"Type of validation_sequences: {type(validation_sequences)}, example: {validation_sequences[:2]}")
print(f"Type of test_sequences: {type(test_sequences)}, example: {test_sequences[:2]}")


assert len(train_sequences) == len(train_labels), "Train sequences and labels length mismatch"
assert len(validation_sequences) == len(validation_labels), "Validation sequences and labels length mismatch"
assert len(test_sequences) == len(test_labels), "Test sequences and labels length mismatch"
print(train_sequences[0])
print(train_labels[0])
print("length of sequence 0", print(len(train_sequences[0])))
print("length of labels 0", print(len(train_labels[0])))


# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
print("alphabet: ", alphabet)
model.to(device)
print(dir(model))
print("batch converter: ", batch_converter)
#config = model.config
#print(config)
model_embed_dim = model.embed_dim
max_length = 1026

print("embedding dimension =", model_embed_dim)
print("max length = ", max_length)

model.eval()

def embed_dataset(model, batch_converter, sequences, max_length=1026):
    inputs_embedding = []
    with torch.no_grad():
        with autocast():
            for i, sample in enumerate(tqdm(sequences)):
                data = [(f"sample_{i}", sample)]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                batch_tokens = batch_tokens.to(device)
                results = model(batch_tokens, repr_layers=[36])
                token_representations = results["representations"][36]
                del results
                l = len(sample)
                L = batch_lens[0].item()
                token_representations = token_representations[0, 1 : L - 1].detach().cpu().numpy()
                inputs_embedding.append(token_representations)
    return inputs_embedding

training_embeddings = embed_dataset(model, batch_converter, train_sequences, max_length)
validation_embeddings = embed_dataset(model, batch_converter, validation_sequences, max_length)
test_embeddings = embed_dataset(model, batch_converter, test_sequences, max_length)

print(training_embeddings[0].shape)
print(validation_embeddings[0].shape)
print(test_embeddings[0].shape)

def pad_labels(labels, max_length):
# Pad labels to length 2048
    padded_labels = []
    for label in labels:
        if len(label) < max_length:
            # Pad the label with zeros
            padded_label = label + [0] * (max_length - len(label))
        else:
            # Truncate the label if it's longer than 1024
            padded_label = label[:max_length]
        padded_labels.append(padded_label)
    return padded_labels

def trunc_labels(labels, max_length=1026):
    truncated_labels = []
    for label in labels:
        if len(label) > max_length:
            truncated_labels.append(label[:max_length])
        else:
            truncated_labels.append(label)
    return truncated_labels


print("len train embeds [0]", len(training_embeddings[0]))
print("len train labels [0]", len(train_labels[0]))



class AlloDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
       # print(f"Encodings length: {len(encodings)}, Labels length: {len(labels)}")  # Debug print

    def __getitem__(self, idx):
        # Ensure idx is valid
        if idx >= len(self.encodings) or idx >= len(self.labels):
            raise IndexError("Index out of range in __getitem__")
        embedding = self.encodings[idx]
        labels = self.labels[idx]
        return {'embed': torch.tensor(embedding), 'labels': torch.tensor(labels)}

    def __len__(self):
        return len(self.labels)



training_dataset = AlloDataset(training_embeddings, train_labels)
validation_dataset = AlloDataset(validation_embeddings, validation_labels)
test_dataset = AlloDataset(test_embeddings, test_labels)

def compute_weights(training_labels):
  total_samples = sum(len(seq) for seq in training_labels)
  class_0_samples = sum(seq.count(0) for seq in training_labels)
  class_1_samples = sum(seq.count(1) for seq in training_labels)

  weight_0 = total_samples / (2 * class_0_samples)
  weight_1 = total_samples / (2 * class_1_samples)
  return weight_0, weight_1
weight_0, weight_1 = compute_weights(train_labels)
print("weights")
print(weight_0, weight_1)

def model_init(num_tokens, embed_dim):
    hidden_dim = int(embed_dim / 2)
    num_hidden_layers = 1 # Number of hidden layers in ConvBert.
    nlayers = 1 # Number of ConvBert layers.
    nhead = 8 # Number of attention heads in ConvBert.
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

# Custom Trainer
class CustomTrainer(Trainer):
    def __init__(self, weight_0, weight_1, **kwargs):
        super().__init__(**kwargs)
        self.weights = torch.tensor([weight_0, weight_1]).float().cuda()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        embeddings = inputs.get("embed").to(device)

        # Print shapes to debug
        #print(f"Embeddings shape: {embeddings.shape}")
        #print(f"Labels shape: {labels.shape}")

        # Forward pass
        outputs = model(embeddings)
        logits = outputs.get("logits")

        #print(logits.shape)
        #print(labels.shape)

        # Compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=self.weights)
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

model_type = 'esm2_t36_3B_UR50D'
experiment = f'ASD_{model_type}'

training_args = TrainingArguments(
    output_dir=f'./results_{experiment}',
    num_train_epochs=40,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=1000,
    learning_rate=2e-04,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_dir=f'./logs_{experiment}',
    logging_steps=200,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=16,
    fp16=True,
    fp16_opt_level="02",
    run_name=experiment,
    seed=seed,
    load_best_model_at_end=True,
    #metric_for_best_model="eval_accuracy",
    metric_for_best_model="eval_F1",
    greater_is_better=True,
    save_strategy="epoch",
    save_total_limit=1
)

training_dataset[0].get("embed").shape
print("embed dim= ", model_embed_dim)
trainer = CustomTrainer(
    weight_0=weight_0,
    weight_1=weight_1,
    model_init=partial(model_init, num_tokens=2, embed_dim=model_embed_dim),
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)


# Train model
trainer.train()

predictions, labels, metrics_output = trainer.predict(test_dataset)

print(metrics_output)

path = '~/PLM_Allosteric_Classification/embedding_extraction/TransferLearning/esm2_t36_650M_UR50D_pdb.ckpt'
torch.save(trainer.model.state_dict(), path)

