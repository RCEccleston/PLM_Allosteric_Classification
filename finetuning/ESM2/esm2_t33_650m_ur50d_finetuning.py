
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import wandb
import numpy as np
import torch
import torch.nn as nn
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    EvalPrediction,
    AutoConfig,
    AutoModel
)
from datasets import Dataset
from accelerate import Accelerator
# Imports specific to the custom peft lora model
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

import evaluate
from sklearn.metrics import average_precision_score
from sklearn import metrics
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
from scipy.special import softmax
import ast
import re
import random
from transformers import AutoModel

import pandas as pd
import numpy as np
import ankh

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

train_file = '/home/lshre1/PredAllo/train_df.csv'
test_file = '/home/lshre1/PredAllo/test_df.csv'
train_df = pd.read_csv(train_file)
test_df_total = pd.read_csv(test_file)
train_sequences = train_df['Sequences'].tolist()
train_labels = train_df['Labels'].apply(ast.literal_eval).tolist()
N_test_total = len(test_df_total)

print(train_labels[0])

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

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
max_sequence_length = 1000

# Helper Functions and Data Preparation
def truncate_labels(labels, max_length):
    """Truncate labels to the specified max_length."""
    return [label[:max_length] for label in labels]

train_tokenized = tokenizer(train_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)
test_tokenized = tokenizer(test_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)
valid_tokenized = tokenizer(validation_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)

# Directly truncate the entire list of labels
train_labels = truncate_labels(train_labels, max_sequence_length)
test_labels = truncate_labels(test_labels, max_sequence_length)
valid_labels = truncate_labels(validation_labels, max_sequence_length)


train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)
valid_dataset = Dataset.from_dict({k: v for k, v in valid_tokenized.items()}).add_column("labels", valid_labels)

print(train_labels[0])

# Compute Class Weights
# Convert classes to a numpy array
classes = np.array([0, 1])
flat_train_labels = [label for sublist in train_labels for label in sublist]
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=flat_train_labels)
accelerator = Accelerator()
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(accelerator.device)

print(class_weights)

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
        "f1": f1,
        "Accuracy": accuracy
    }

def compute_loss(model, inputs):
    """Custom compute_loss function."""
    logits = model(**inputs).logits
    labels = inputs["labels"]
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    active_loss = inputs["attention_mask"].view(-1) == 1
    active_logits = logits.view(-1, model.config.num_labels)
    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
    )
    loss = loss_fct(active_logits, active_labels)
    return loss

# Define Custom Trainer Class
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = compute_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss

class CustomModelForTokenClassification(AutoModelForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        # Replace the classification head with Ankh's head
        # Initialize Ankh's ConvBERT classification head
        hidden_size = 1280
        num_labels = 2
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


    def forward(self, **kwargs):
        # Forward pass with the new classifier head
        outputs = super().forward(**kwargs)
        # Use your custom head to predict logits
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)  # Custom classification head
        return (logits,) + outputs[1:]  # Return the logits along with the rest of the outputs (if any)

def train_function_no_sweeps(train_dataset, valid_dataset, test_dataset):

    # Set the LoRA config
    config = {
        "lora_alpha": 1, #try 0.5, 1, 2, ..., 16
        "lora_dropout": 0.2,
        "lr": 5.701568055793089e-04,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.5,
        "num_train_epochs": 40,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "r": 2,
        "weight_decay": 0.2,
        # Add other hyperparameters as needed
    }
    # The base model you will train a LoRA on top of
    model_checkpoint = "facebook/esm2_t33_650M_UR50D"

    # Define labels and model
    id2label = {0: "No binding site", 1: "Binding site"}
    label2id = {v: k for k, v in id2label.items()}
    #model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    # Load the base ESM2 model without the classification head
    #model = AutoModel.from_pretrained(model_checkpoint)

   # Load the pre-trained model config and then initialize the custom model

    #config = AutoConfig.from_pretrained(model_checkpoint)

    # Define number of labels for your token classification task
    num_labels = 2  # or however many classes you have

    # Initialize your custom model with the overridden classification head
    model = CustomModelForTokenClassification.from_pretrained(model_checkpoint)


    # Convert the model into a PeftModel
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["query", "key", "value", "dense_h_to_4h", "dense_4h_to_h"], # also try "dense_h_to_4h" and "dense_4h_to_h"
        lora_dropout=config["lora_dropout"],
        bias="none" # or "all" or "lora_only"
    )
    model = get_peft_model(model, peft_config)

    # Use the accelerator
    model = accelerator.prepare(model)
    train_dataset = accelerator.prepare(train_dataset)
    test_dataset = accelerator.prepare(test_dataset)
    valid_dataset = accelerator.prepare(valid_dataset)


    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Training setup

    training_args = TrainingArguments(
        output_dir=f"esm2_t33_650M-lora-binding-sites_{timestamp}",
        learning_rate=config["lr"],
        lr_scheduler_type=config["lr_scheduler_type"],
        gradient_accumulation_steps=5,
        max_grad_norm=config["max_grad_norm"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir="./logs",  # Ensure this is set
        logging_first_step=True,  # Ensure logging happens on the first step
        logging_steps=50,  # Adjust for your dataset size
        save_total_limit=7,
        no_cuda=False,
        seed=8893,
        fp16=True,
        report_to="wandb"
    )


    # Initialize Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # Train and Save Model
    trainer.train()
    save_path = os.path.join("lora_binding_sites", f"best_model_esm2_t33_650M_lora_{timestamp}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    eval_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("Evaluation Metrics:", eval_metrics)

train_function_no_sweeps(train_dataset, valid_dataset, test_dataset)

