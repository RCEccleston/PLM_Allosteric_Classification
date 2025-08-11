
#import dependencies
import os.path
#os.chdir("set a path here")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import re
import numpy as np
import pandas as pd
import copy

import transformers, datasets
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import T5EncoderModel, T5Tokenizer
from transformers import TrainingArguments, Trainer, set_seed
from transformers import DataCollatorForTokenClassification

from evaluate import load
import datasets

from tqdm import tqdm
import random

from scipy import stats
from sklearn import metrics
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import ast

import ankh
from functools import partial

# Set environment variables to run Deepspeed from a notebook
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9993"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

print("Torch version: ",torch.__version__)
print("Cuda version: ",torch.version.cuda)
print("Numpy version: ",np.__version__)
print("Pandas version: ",pd.__version__)
print("Transformers version: ",transformers.__version__)
print("Datasets version: ",datasets.__version__)

print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return the number of GPUs
print(torch.cuda.current_device())  # Should return the index of the current GPU
print(torch.cuda.get_device_name(0))  # Should return the name of the GPU

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print('Available device:', device)

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

# Check the types of the split sequences
print(f"Type of train_sequences: {type(train_sequences)}, example: {train_sequences[:2]}")
print(f"Type of validation_sequences: {type(validation_sequences)}, example: {validation_sequences[:2]}")
print(f"Type of test_sequences: {type(test_sequences)}, example: {test_sequences[:2]}")

assert len(train_sequences) == len(train_labels), "Train sequences and labels length mismatch"
assert len(validation_sequences) == len(validation_labels), "Validation sequences and labels length mismatch"
assert len(test_sequences) == len(test_labels), "Test sequences and labels length mismatch"

my_train = {"sequence" : train_sequences, "label" : train_labels}
my_train = pd.DataFrame(my_train)
my_valid = {"sequence" : validation_sequences, "label" : validation_labels}
my_valid = pd.DataFrame(my_valid)
my_test = {"sequence" : test_sequences, "label" : test_labels}
my_test = pd.DataFrame(my_test)

# Set gpu device
gpu=1
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu-1)
# Set the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

half_precision = False
if not half_precision:
  model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
  tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
elif half_precision and torch.cuda.is_available() :
  tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
  model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16).to(torch.device('cuda'))
else:
  raise ValueError('Half precision can be run on GPU only.')

# Move model to GPU
model.to(device)

sequence_examples = ["PRTEINO", "SEQWENCE"]
# this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

# generate embeddings
with torch.no_grad():
    embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)

# extract embeddings for the first ([0,:]) sequence in the batch while removing padded & special tokens ([0,:7])
emb_0 = embedding_repr.last_hidden_state[0,:7] # shape (7 x 1024)
print(f"Shape of per-residue embedding of first sequences: {emb_0.shape}")
# do the same for the second ([1,:]) sequence in the batch while taking into account different sequence lengths ([1,:8])
emb_1 = embedding_repr.last_hidden_state[1,:8] # shape (8 x 1024)

# if you want to derive a single representation (per-protein embedding) for the whole protein
emb_0_per_protein = emb_0.mean(dim=0) # shape (1024)

print(f"Shape of per-protein embedding of first sequences: {emb_0_per_protein.shape}")

outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
embedding = outputs[0].detach().squeeze().cpu().numpy()
print(f"Shape of per-residue embedding of first sequences: {embedding[0].shape}")
emdedding_0 = embedding[0,:7,:]
print(f"Shape of per-residue embedding of first sequences: {emdedding_0.shape}")

print(embedding_repr.last_hidden_state.shape)

# Preprocess inputs
print("Preprocess inputs")
my_train["sequence"] = my_train["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
my_valid["sequence"] = my_valid["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
my_test["sequence"] = my_test["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)

# Add spaces between each amino acid for PT5 to correctly use them
my_train['sequence'] = my_train.apply(lambda row: " ".join(row["sequence"]), axis=1)
my_valid['sequence'] = my_valid.apply(lambda row: " ".join(row["sequence"]), axis=1)
my_test['sequence'] = my_test.apply(lambda row: " ".join(row["sequence"]), axis=1)

def embed_dataset(model,tokenizer,seqs,labels):
    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    reprs = []
    embeddings = []
    batch = 32
    N_batches = int(np.ceil(len(seqs)/batch))
    # generate embeddings
    with torch.no_grad():
      for i in range(N_batches):
        start = i*batch
        end = (i+1)*batch
        input_id = input_ids[start:end]
        mask = attention_mask[start:end]
        embedding_repr = model(input_ids=input_id,attention_mask=mask)
        reprs.append(embedding_repr)
    # Concatenate along dimension 0 (rows)
    concatenated_reprs = torch.cat(reprs, dim=0)

    for i, seq in enumerate(seqs):

      # extract embeddings for the first ([0,:]) sequence in the batch while removing padded & special tokens
      emb = concatenated_reprs[i].last_hidden_state[i,:len(seq)] # shape (len(seq) x 1024)
      print(f"Shape of per-residue embedding: {emb.shape}")
      embeddings.append(emb)

      return embeddings


# Dataset creation function
def create_dataset(tokenizer, seqs, labels):
    # Tokenize the sequences
    tokenized = tokenizer(seqs, max_length=1024, padding='max_length', truncation=True, return_tensors="pt")

    # Pad labels to length 1024
    trunc_labels = []
    for label in labels:
        if len(label) < 1024:

            trunc_label = label
        else:
            # Truncate the label if it's longer than 1024
            trunc_label = label[:1024]
        trunc_labels.append(trunc_label)

    # Add the padded labels to the tokenized dataset
    tokenized["labels"] = trunc_labels

    # Create the dataset from the tokenized data
    dataset = datasets.Dataset.from_dict(tokenized)

    return dataset, trunc_labels

# Preprocess inputs
print("Preprocess inputs")
my_train["sequence"] = my_train["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
my_valid["sequence"] = my_valid["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
my_test["sequence"] = my_test["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)


# Add spaces between each amino acid for PT5 to correctly use them
my_train['sequence'] = my_train.apply(lambda row: " ".join(row["sequence"]), axis=1)
my_valid['sequence'] = my_valid.apply(lambda row: " ".join(row["sequence"]), axis=1)
my_test['sequence'] = my_test.apply(lambda row: " ".join(row["sequence"]), axis=1)

# Create Datasets
print("Creating datasets")
train_set, train_trunc_labels = create_dataset(tokenizer, list(my_train['sequence']), list(my_train['label']))
train_set.set_format("torch")
valid_set, valid_trunc_labels = create_dataset(tokenizer, list(my_valid['sequence']), list(my_valid['label']))
valid_set.set_format("torch")
test_set, test_trunc_labels  = create_dataset(tokenizer, list(my_test['sequence']), list(my_test['label']))
test_set.set_format("torch")

print(train_set)

print(len(train_sequences[0]))

l = train_set["labels"]
 print(len(l[0]))

def embed_dataset(model, dataset):
  embeddings = []
  for input in dataset:
    l = len(input.get("labels"))
    attention_mask = input.get("attention_mask")
    attention_mask = attention_mask.reshape(1,attention_mask.size(0)).to(device)
    #print(attention_mask)
    input_ids=input.get("input_ids")
    input_ids = input_ids.reshape(1, input_ids.size(0)).to(device)
    input_shape = input_ids.size()
    batch_size, seq_length = input_shape
    outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    embedding = outputs[0].detach().squeeze().cpu().numpy()
    embedding = embedding[:l, :]
    embeddings.append(embedding)

  return embeddings

train_embeddings = embed_dataset(model, train_set)
valid_embeddings = embed_dataset(model, valid_set)
test_embeddings = embed_dataset(model, test_set)

train_embeddings[0].shape

train_embeddings[0].squeeze().shape

id2tag = {0: "0", 1: "1"}
tag2id = {"0": 0, "1":1}
def encode_tags(labels):
    labels = [[tag2id[tag] for tag in doc] for doc in labels]
    return labels


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

len(train_labels[1])

training_dataset = AlloDataset(train_embeddings, train_trunc_labels)
validation_dataset = AlloDataset(valid_embeddings, valid_trunc_labels)
test_dataset = AlloDataset(test_embeddings, test_trunc_labels)

training_dataset[0].get("embed").shape

training_dataset[0].get("labels").shape

def compute_weights(training_labels):
  total_samples = sum(len(seq) for seq in training_labels)
  class_0_samples = sum(seq.count(0) for seq in training_labels)
  class_1_samples = sum(seq.count(1) for seq in training_labels)

  weight_0 = total_samples / (2 * class_0_samples)
  weight_1 = total_samples / (2 * class_1_samples)
  return weight_0, weight_1
weight_0, weight_1 = compute_weights(my_train['label'])
print(weight_0, weight_1)

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

def compute_metrics(pred):
    predictions, labels = pred

    # Convert predictions and labels to tensors (if not already)
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)

    # Calculate softmax probabilities along the appropriate dimension
    softmax_predictions = torch.nn.functional.softmax(predictions, dim=-1)

    # Flatten predictions and labels
    flat_predictions = softmax_predictions.view(-1, softmax_predictions.shape[-1])
    flat_labels = labels.view(-1)

    # Filter out the padding labels
    active_loss = flat_labels != -100
    active_predictions = flat_predictions[active_loss]
    active_labels = flat_labels[active_loss]

    # Calculate probabilities for the positive class
    pos_probs = active_predictions[:, 1].cpu().numpy()  # Assuming class 1 is your positive class

    # Binarize predictions for calculating other metrics
    threshold = 0.5
    preds = (pos_probs >= threshold).astype(int)
    active_labels_np = active_labels.cpu().numpy()

    # Calculate metrics
    auc = roc_auc_score(active_labels_np, pos_probs)
    aps = average_precision_score(active_labels_np, pos_probs)
    precision = precision_score(active_labels_np, preds)
    recall = recall_score(active_labels_np, preds)
    f1 = f1_score(active_labels_np, preds)
    accuracy = accuracy_score(active_labels_np, preds)

    return {
        'AUC': auc,
        'APS': aps,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy
    }

mismatch = [(thing.get("embed").shape[0], thing.get("labels").shape[0]) for thing in training_dataset if thing.get("embed").shape[0] != thing.get("labels").shape[0]]

class CustomTrainer(Trainer):
    def __init__(self, weight_0, weight_1, **kwargs):
        super().__init__(**kwargs)
        self.weights=torch.tensor([weight_0, weight_1]).float().cuda()
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # Add num_items_in_batch argument
        labels = inputs.get("labels")
        # forward pass
        outputs = model(inputs.get("embed"))
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=self.weights)
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

model_type = 'ProtBert'
experiment = f'ASD_{model_type}'

training_args = TrainingArguments(
    output_dir=f'drive/MyDrive/results_{experiment}',
    num_train_epochs=40,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=1000,
    learning_rate=1e-04,
    weight_decay=0.0,
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
    #metric_for_best_model="eval_accuracy",
    metric_for_best_model="eval_APS",
    greater_is_better=True,
    save_strategy="epoch"
)

model_embed_dim = 1024 # Embedding dimension for ProtBert large.

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
metrics_output

