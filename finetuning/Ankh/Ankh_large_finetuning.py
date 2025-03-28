''' this script is adapted from https://huggingface.co/blog/AmelieSchreiber/esmbind'''

import ankh
import torch
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification

from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import random
from transformers import set_seed

import pandas as pd
import ast

from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_score, recall_score, f1_score, accuracy_score

from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset

from transformers.modeling_outputs import TokenClassifierOutput

from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Available device:', device)

# Load Ankh2 model and tokenizer
base_model, tokenizer = ankh.load_large_model()

ankh_config = base_model.config
hidden_size = ankh_config.hidden_size
max_length = 512

base_model = base_model.to(device)

print(ankh_config)

print(hidden_size)

def make_mask(sequences):
  masks = []
  for seq in sequences:
    masks.append([1]*len(seq))
  return masks

# Set random seeds for reproducibility of your trainings run
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)


# Set all random seeds
seed= 42      #random seed
set_seeds(seed)

train_file = 'drive/MyDrive/train_df.csv'
test_file = 'drive/MyDrive/test_df.csv'
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

def segment_sequences_with_sliding_windows(sequences, labels, window_size=512, step_size=128):
    """
    Segment a list of sequences and their corresponding labels into sliding overlapping windows.

    Args:
        sequences (list of str): List of protein sequences to segment.
        labels (list of list of int): List of corresponding labels (same length as sequences).
        window_size (int): Size of each sliding window.
        step_size (int): Step size for the sliding window.

    Returns:
        segmented_sequences (list of str): List of segmented sequences.
        segmented_labels (list of list of int): List of segmented labels corresponding to sequences.
    """
    segmented_sequences = []
    segmented_labels = []

    for seq, lbl in zip(sequences, labels):
        # Ensure the sequence and label lengths match
        assert len(seq) == len(lbl), "Sequence and label length mismatch!"

        # Segment the sequence and labels using sliding windows
        for start_idx in range(0, len(seq), step_size):
            end_idx = start_idx + window_size

            # Extract windowed sequences and labels
            window_seq = seq[start_idx:end_idx]
            window_lbl = lbl[start_idx:end_idx]

            # If the window is smaller than window_size (e.g., at the end), pad
            if len(window_seq) < window_size:
                # Pad the sequence with "X" and labels with 0
                window_seq += "X" * (window_size - len(window_seq))
                window_lbl += [0] * (window_size - len(window_lbl))

            # Add the segmented window to the list
            segmented_sequences.append(window_seq)
            segmented_labels.append(window_lbl)

    return segmented_sequences, segmented_labels


window_size = max_length
step_size = 256

#segmented_sequences, segmented_labels = segment_with_sliding_windows(sequence, labels, window_size, step_size)

seg_train_sequences, seg_train_labels = segment_sequences_with_sliding_windows(train_sequences, train_labels, window_size, step_size)
seg_test_sequences, seg_test_labels = segment_sequences_with_sliding_windows(test_sequences, test_labels, window_size, step_size)
seg_validation_sequences, seg_validation_labels = segment_sequences_with_sliding_windows(validation_sequences, validation_labels, window_size, step_size)

print(len(train_sequences))
print(len(seg_train_sequences))

train_masks = make_mask(seg_train_sequences)
validation_masks = make_mask(seg_validation_sequences)
test_masks = make_mask(seg_test_sequences)

#def preprocess_dataset(sequences, labels, max_length=None):
#    sequences = sequences.astype(str)
#    sequences = ["".join(seq.split()) for seq in sequences]

#    if max_length is None:
#        max_length = len(max(sequences, key=lambda x: len(x)))

#    seqs = [list(seq)[:max_length] for seq in sequences]
#    labels = [list(label)[:max_length] for label in labels]
#    labels = [[str(num) for num in label] for label in labels]


#    assert len(seqs) == len(labels)
#    return seqs, labels

def preprocess_dataset(sequences, labels, max_length=None):
    # Convert all elements in the list to strings
    sequences = [str(seq) for seq in sequences]
    sequences = ["".join(seq.split()) for seq in sequences]

    if max_length is None:
        max_length = len(max(sequences, key=lambda x: len(x)))

    seqs = [list(seq)[:max_length] for seq in sequences]
    labels = [list(label)[:max_length] for label in labels]
    labels = [[int(num) for num in label] for label in labels]

    assert len(seqs) == len(labels)
    return seqs, labels

def embed_dataset(model, sequences, shift_left = 0, shift_right = -1):
    inputs_embedding = []
    with torch.no_grad():
        for sample in tqdm(sequences):
            ids = tokenizer.batch_encode_plus([sample], add_special_tokens=True,
                                              padding=True, is_split_into_words=True,
                                              return_tensors="pt")
            embedding = model(input_ids=ids['input_ids'].to(device))[0]
            embedding = embedding[0].detach().cpu().numpy()[shift_left:shift_right]
            inputs_embedding.append(embedding)
    return inputs_embedding


from torch import tensor

def tokenize_sequences(tokenizer, sequences, shift_left=0, shift_right=-1):
    input_ids = []

    for sample in tqdm(sequences):
        # Tokenize the sequence and return tensors
        ids = tokenizer.batch_encode_plus(
            [sample],  # Always pass a list of sequences, even if it's just one
            add_special_tokens=True,
            max_length=max_length,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt"
        )
        ids = ids['input_ids']  # Extract input_ids tensor
        print(ids.shape)

        # Handle shift_left and shift_right by slicing along the sequence dimension
        ids = ids[:, shift_left:shift_right]  # Slice along the sequence dimension

        input_ids.append(ids)  # Append the tensor to the list

    # Concatenate the list of tensors along the batch dimension (dim=0)
    return torch.cat(input_ids, dim=0)  #

# Dataset creation
def create_dataset(tokenizer,seqs,labels, max_length):
    tokenized = tokenizer(seqs, max_length=max_length, padding=True, add_special_tokens=True, is_split_into_words=True, return_tensors="pt")
    dataset = Dataset.from_dict(tokenized)
    # we need to cut of labels after 1023 positions for the data collator to add the correct padding (1023 + 1 special tokens)
    labels = [l[:max_length] for l in labels]
    dataset = dataset.add_column("labels", labels)

    return dataset

train_sequences, train_labels = preprocess_dataset(seg_train_sequences, seg_train_labels, max_length)
validation_sequences, validation_labels = preprocess_dataset(seg_validation_sequences, seg_validation_labels, max_length)
test_sequences, test_labels = preprocess_dataset(seg_test_sequences, seg_test_labels, max_length)

print(len(train_sequences))
print(len(train_labels))

#train_ids = tokenize_sequences(tokenizer, train_sequences, shift_left=0, shift_right=-1)
#validation_ids = tokenize_sequences(tokenizer, validation_sequences, shift_left=0, shift_right=-1)
#test_ids = tokenize_sequences(tokenizer, test_sequences, shift_left=0, shift_right=-1)

my_train = {"sequence" : train_sequences, "labels" : train_labels, "mask" : train_masks}
my_train = pd.DataFrame(my_train)
my_valid = {"sequence" : validation_sequences, "labels" : validation_labels, "mask" : validation_masks}
my_valid = pd.DataFrame(my_valid)
my_test = {"sequence" : test_sequences, "labels" : test_labels, "mask" : test_masks}
my_test = pd.DataFrame(my_test)

# Replace uncommon AAs with "X"
#my_train["sequence"]=my_train["sequence"].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
#my_valid["sequence"]=my_valid["sequence"].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
#my_test["sequence"]=my_test["sequence"].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)

# Create Datasets
#train_set=create_dataset(tokenizer,list(my_train["sequence"]),list(my_train['label']), max_length)
#valid_set=create_dataset(tokenizer,list(my_valid['sequence']),list(my_valid['label']), max_length)
#test_set=create_dataset(tokenizer,list(my_test['sequence']),list(my_test['label']), max_length)

train_set = Dataset.from_dict(my_train)
valid_set = Dataset.from_dict(my_valid)
test_set = Dataset.from_dict(my_test)

def tokenize_and_pad(example, tokenizer):
  tokens = tokenizer.batch_encode_plus([example['sequence']], is_split_into_words=True, add_special_tokens=True)
  example['input_ids'] = tokens['input_ids'][0][:-1].copy()
  example['attention_mask'] = tokens['attention_mask'][0][:-1].copy()


  return example

train_set = train_set.map(lambda x: tokenize_and_pad(x, tokenizer)).remove_columns(['sequence', 'mask'])
valid_set = valid_set.map(lambda x: tokenize_and_pad(x, tokenizer)).remove_columns(['sequence', 'mask'])
test_set = test_set.map(lambda x: tokenize_and_pad(x, tokenizer)).remove_columns(['sequence', 'mask'])

train_set.set_format("torch")
valid_set.set_format("torch")
test_set.set_format("torch")

classes = np.array([0, 1])
flat_train_labels = [label for sublist in train_labels for label in sublist]
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=flat_train_labels)

class Ankh2ForTokenClassification(nn.Module):
    def __init__(self, base_model, config, num_labels=2, class_weights=None):
        super().__init__()
        self.config = config
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(base_model.device)
        else:
            self.class_weights = None
        self.base_model = base_model  # Ankh2 model
        # Enable output_hidden_states
        self.base_model.config.output_hidden_states = True
        hidden_size = 1536
        self.num_labels = num_labels
        hidden_dim = int(hidden_size / 2)
        num_hidden_layers = 1 # Number of hidden layers in ConvBert.
        nlayers = 1 # Number of ConvBert layers.
        nhead = 4
        dropout = 0.2
        conv_kernel_size = 7


        self.classifier = ankh.ConvBertForMultiClassClassification(num_tokens=num_labels,
                                                                input_dim=hidden_size,
                                                                nhead=nhead,
                                                                hidden_dim=hidden_dim,
                                                                num_hidden_layers=num_hidden_layers,
                                                                num_layers=nlayers,
                                                                kernel_size=conv_kernel_size,
                                                                dropout=dropout)


    def forward(self, input_ids=None, attention_mask=None, labels=None, input_embeds=None, **kwargs):

      # Get model outputs from the base model (e.g., T5)
      #outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
      outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
      #if input_ids is not None:
      #    outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
      #elif input_embeds is not None:
      #    outputs = self.base_model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True)
      #else:
      #    raise ValueError("You have to specify either input_ids or inputs_embeds")

      # Depending on the model type, the output might be a tuple (not a dictionary).
      print(outputs)
      if isinstance(outputs, tuple):
        hidden_states = outputs[0]  # First element is often the hidden states
      else:
        hidden_states = outputs['last_hidden_state']
      #print(f"Hidden states: {hidden_states}")
      # Pass hidden states through the classifier
      logits = self.classifier(hidden_states)
      if isinstance(logits, TokenClassifierOutput):
        logits = logits.logits  # Extract the tensor
      #print(f"Logits type: {type(logits)}")  # Should be <class 'torch.Tensor'>
      #print(f"Logits shape: {logits.shape}")
      # Calculate loss if labels are provided
      loss = None
      if labels is not None:
          labels = labels.to(self.base_model.device)
          #print(f"Labels type: {type(labels)}
          # CrossEntropyLoss expects class indices as labels, not one-hot
          loss_fn = nn.CrossEntropyLoss(weight=self.class_weights) if self.class_weights is not None else nn.CrossEntropyLoss()

          # Debugging: Ensure labels are a tensor
          #print(f"Labels type: {type(labels)}")  # Should be <class 'torch.Tensor'>
          #print(f"Labels shape: {labels.shape}")

          # Compute loss
          loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

      # Return output as TokenClassifierOutput
      return TokenClassifierOutput(
          loss=loss,
          logits=logits,  # Pass the tensor logits here
          hidden_states=hidden_states,
          attentions=outputs.attentions,
      )


def compute_metrics(pred):
      predictions, labels = pred
      #print("predictions = ", predictions)
      #print("labels = ", labels)
      # Convert predictions and labels to tensors (if not already)
      #predictions = torch.tensor(predictions.logits)
      predictions = pred.predictions  # This will return the tuple of (logits, labels)
      #print(predictions)
      logits = predictions[1]  # Access the logits in the second element of the tuple
      #print("Logits (Sample):", logits[:5])  # Print logits for the first few samples

      labels = pred.label_ids

      # Convert logits and labels to tensors (if they are not already)
      logits = torch.tensor(logits)
      labels = torch.tensor(labels) # Ensure labels is a tensor
      #print(f"logits shape: {logits.shape}")

      # Calculate softmax probabilities along the appropriate dimension
      softmax_predictions = torch.nn.functional.softmax(logits, dim=-1)
      #print(f"softmax_predictions shape: {softmax_predictions.shape}")
      #print("Softmax probabilities (Sample):", softmax_predictions[:5])

      # Flatten predictions and labels
      # flat_predictions = softmax_predictions.reshape(-1, softmax_predictions.shape[-1])
      # flat_labels = labels.reshape(-1)
      # Flatten predictions and labels
      flat_predictions = torch.argmax(softmax_predictions, dim=-1).flatten()
      #print(f"flat_predictions shape: {flat_predictions.shape}")

      flat_labels = labels.flatten()

      # Check if the sizes match before applying the mask
      if flat_labels.shape != flat_predictions.shape:
        print(f"Shape mismatch: flat_labels shape = {flat_labels.shape}, flat_predictions shape = {flat_predictions.shape}")
        raise ValueError("Shape mismatch between labels and predictions")

      # Filter out the padding labels
      active_loss = flat_labels != -100
      #print(f"active_loss shape: {active_loss.shape}")

      active_predictions = flat_predictions[active_loss]
      active_labels = flat_labels[active_loss]


      # Print to check for positives in labels and predictions
      print("Flat Predictions contain positives (1):", (active_predictions == 1).any().item())
      print("Flat Labels contain positives (1):", (active_labels == 1).any().item())
      print("Number of positives in Predictions:", (active_predictions == 1).sum().item())
      print("Number of positives in Labels:", (active_labels == 1).sum().item())


      # Calculate probabilities for the positive class
      pos_probs = active_predictions.cpu().numpy()  # Assuming class 1 is your positive class

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

print(class_weights)

# Set the LoRA config
config = {
    "lora_alpha": 8, #try 0.5, 1, 2, ..., 16
    "lora_dropout": 0.2,
    "lr": 1e-5,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 0.5,
    "num_train_epochs": 20,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "r": 4,
    "weight_decay": 0.2,
    # Add other hyperparameters as needed
    }
# Convert the model into a PeftModel
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=config["r"],
    lora_alpha=config["lora_alpha"],
    target_modules=["query", "key", "value"], # also try "dense_h_to_4h" and "dense_4h_to_h"
    lora_dropout=config["lora_dropout"],
    bias="none" # or "all" or "lora_only"
    )
# Wrap the model with LoRA
model = Ankh2ForTokenClassification(base_model=base_model, config=ankh_config, num_labels=2, class_weights=class_weights)
model = get_peft_model(model, peft_config)
model = model.to(device)

for name, param in model.named_parameters():
    if "lora" in name:  # Replace with the specific name pattern for LoRA layers
        print(name, param.mean().item(), param.std().item())

if param.requires_grad and torch.isnan(param).any():
    torch.nn.init.normal_(param, mean=0.0, std=0.02)

import torch
import torch.nn as nn
import torch.nn.init as init

# Function to initialize weights
def initialize_lora_layers(model):
    for name, param in model.named_parameters():
        # Target LoRA layers (you can adapt this if necessary based on your layer naming convention)
        if 'lora' in name.lower():
            if 'lora_A' in name:
                init.xavier_normal_(param)  # Xavier Normal Initialization
            elif 'lora_B' in name:
                init.kaiming_normal_(param)  # Kaiming Normal Initialization
            else:
                init.normal_(param, mean=0.0, std=0.02)  # Default Normal Initialization

            print(f"Initialized {name} with custom initialization.")

# Example of applying it to your PEFT model
initialize_lora_layers(model)  # Assuming `peft_model` is the instantiated PEFT model

#print_weights(model)

for name, param in base_model.named_parameters():
    if 'lora' in name.lower():  # Adjust search term if needed
        print(f"LoRA Layer Found: {name}")

for name, param in base_model.named_parameters():
    if 'lora' in name.lower():  # Lowercasing for case-insensitive search
        print(f"LoRA Layer Found: {name}")

for name, param in model.named_parameters():
    if 'lora' in name.lower():  # Check in the PEFT model
        print(f"LoRA Layer Found in PEFT: {name}")

# Store the initial weights before applying LoRA for specific layers
initial_weights = {}
target_layers = ["query", "key", "value", "intermediate.dense", "output.dense"]  # List of layers you are modifying with LoRA

# Save initial weights of the target layers
for name, param in base_model.named_parameters():
    if any(layer_name in name for layer_name in target_layers):
        initial_weights[name] = param.data.clone()

# Apply LoRA
#model = get_peft_model(model, peft_config)
#model = model.to(device)

# Compare initial and LoRA-modified weights
for name, param in model.named_parameters():
    if name in initial_weights:
        print(f"Layer: {name}, Weight diff min: {(param.data - initial_weights[name]).min()}, Weight diff max: {(param.data - initial_weights[name]).max()}")

print("Model device:", next(model.parameters()).device)

lr=3e-4
batch = 1
accum = 8
epochs = 20
args = TrainingArguments(
    output_dir="./ankh_finetuned",
    evaluation_strategy = "epoch",
    eval_steps =500,
    logging_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=lr,
    lr_scheduler_type="cosine_with_restarts",
    per_device_train_batch_size=batch,
    per_device_eval_batch_size=batch,
    gradient_accumulation_steps=accum,
    num_train_epochs=epochs,
    seed = seed,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    greater_is_better=True,
    metric_for_best_model="F1",
    #no_cuda=False,
    fp16=False
    #fp16=torch.cuda.is_available(),  # Enable mixed precision if using GPU
    )

# Store the initial weights before applying LoRA for specific layers
initial_weights = {}
target_layers = ["query", "key", "value", "dense_h_to_4h", "dense_4h_to_h", "lora_A", "lora_B"]

# Save initial weights of the target layers, including LoRA layers
for name, param in base_model.named_parameters():
    if any(layer_name in name for layer_name in target_layers):
        print(f"Saving initial weights for {name}")
        initial_weights[name] = param.data.clone()

# Apply LoRA
#model = get_peft_model(model, peft_config)
#model = model.to(device)

# Print out the layers in the LoRA-modified model to check names
#for name, param in model.named_parameters():
#    print(f"Layer: {name}")

# Compare initial and LoRA-modified weights
for name, param in model.named_parameters():
    if any(layer_name in name for layer_name in target_layers):  # Match based on substrings
        if name in initial_weights:
            print(f"Layer: {name}, Weight diff min: {(param.data - initial_weights[name]).min()}, Weight diff max: {(param.data - initial_weights[name]).max()}")
        else:
            print(f"Layer: {name} has no initial weights stored.")

# Before applying LoRA, print the initial weights
for name, param in base_model.named_parameters():
    if any(layer_name in name for layer_name in target_layers):
        print(f"Initial weights for {name}: min: {param.data.min()}, max: {param.data.max()}")

# After applying LoRA, print the modified weights
for name, param in model.named_parameters():
    if name in initial_weights:
        print(f"Modified weights for {name}: min: {param.data.min()}, max: {param.data.max()}")

base_model.named_parameters()

#train_set.features

from transformers import TrainerCallback

class GradientMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Check for NaN in gradients after each step
        for name, param in kwargs['model'].named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of {name}")
        return control

# Create a custom trainer with the gradient monitoring callback
gradient_monitor = GradientMonitorCallback()

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=valid_set,
    compute_metrics=compute_metrics,
    callbacks=[gradient_monitor]  # Adding the callback to monitor gradients

    )

trainer.train()

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} requires gradients.")

