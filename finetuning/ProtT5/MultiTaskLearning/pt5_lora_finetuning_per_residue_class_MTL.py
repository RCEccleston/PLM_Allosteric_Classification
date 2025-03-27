"""## Imports and env. variables"""

#import dependencies
import os.path
#os.chdir("set a path here")
import wandb
import pickle
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader

import re
import numpy as np
import pandas as pd
import copy

import transformers, datasets
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import T5EncoderModel, T5Tokenizer, TrainerCallback



from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AdamW,
    set_seed,
    EvalPrediction
)
from evaluate import load
from datasets import Dataset

from tqdm import tqdm
import random

from scipy import stats
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import ankh

from accelerate import Accelerator
# Imports specific to the custom peft lora model
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

import ast
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_score, recall_score, f1_score, accuracy_score

"""0e74b2cfa4ea8ef797914d7bf53118de17eb4e82

1.   List item
2.   List item



1.   List item
2.   List item


"""

from accelerate import Accelerator
import wandb

wandb.init(project='my-project')

from torch.optim import AdamW  # Import AdamW from torch.optim

import logging
logging.basicConfig(level=logging.DEBUG)

# Set environment variables to run Deepspeed from a notebook
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9993"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

"""# Environment to run this notebook


These are the versions of the core packages we use to run this notebook:
"""

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

"""**For easy setup of this environment you can use the finetuning.yml File provided in this folder**

check here for [setting up env from a yml File](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

# Input data

Provide your training and validation data in seperate pandas dataframes

example shown below

**Modify the data loading part above as needed for your data**

To run the training you need two dataframes (training and validation) each with the columns "sequence" and "label" and "mask"

Columns are:
+ protein sequence
+ label is a list of len(protein sequence) with integers (from 0 to number of classes - 1) corresponding to predicted class at this position
+ mask gives the possibility to ignore parts of the positions. Provide a list of len(protein sequence) where 1 is processed, while 0 is ignored
"""

def make_mask(sequences):
  masks = []
  for seq in sequences:
    masks.append([1]*len(seq))
  return masks

HALF_PRECISION = True

train_file = '/home/lshre1/PredAllo/train_df_ssp.csv'
test_file = '/home/lshre1/PredAllo/test_df_ssp.csv'
train_df = pd.read_csv(train_file)
test_df_total = pd.read_csv(test_file)
train_sequences = train_df['Sequences'].tolist()
train_labels = train_df['Labels'].apply(ast.literal_eval).tolist()
N_test_total = len(test_df_total)
train_ssp = train_df['Secondary_Structure'].apply(ast.literal_eval)
print(train_ssp[0])
s = train_ssp[0]
print(s[0])
DSSP_MAP = {'H': 0, 'E': 1, 'C': 2}

#[print(label) for label in protein_labels for protein_labels in train_ssp]
train_ssp_labels = [[DSSP_MAP[label] for label in protein_labels] for protein_labels in train_ssp]

print("len train_sequences_0 = ", len(train_sequences[0]))
print("len train_labels_0 = ", len(train_labels[0]))
print("len train_ssp_0 = ", len(train_ssp_labels[0]))

print("checking label lengths")
for i, (lbl, ssp_lbl) in enumerate(zip(train_labels, train_ssp_labels)):
#    print(len(lbl), len(ssp_lbl))
    if len(lbl) !=len(ssp_lbl):
        print(f"Mismatch at index {i}: labels has {len(lbl)}, ssp_labels has {len(ssp_lbl)}")




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
test_ssp = test_df['Secondary_Structure'].apply(ast.literal_eval)
test_ssp_labels = [[DSSP_MAP[label] for label in protein_labels] for protein_labels in test_ssp]


validation_sequences = valid_df['Sequences'].tolist()
validation_labels = valid_df['Labels'].apply(ast.literal_eval).tolist()
validation_ssp = valid_df['Secondary_Structure'].apply(ast.literal_eval)
validation_ssp_labels = [[DSSP_MAP[label] for label in protein_labels] for protein_labels in validation_ssp]


print(len(train_sequences))
print(len(validation_sequences))
print(len(test_sequences))


print(len(train_sequences), len(train_labels), len(train_ssp_labels))
assert len(train_sequences) == len(train_labels), "Train sequences and labels length mismatch"
assert len(validation_sequences) == len(validation_labels), "Validation sequences and labels length mismatch"
assert len(test_sequences) == len(test_labels), "Test sequences and labels length mismatch"
assert len(train_sequences) == len(train_ssp_labels), "Train sequences and ssp labels length mismatch"
assert len(train_labels) == len(train_ssp_labels), "Train labels and Train ssp labels length mismatch"
# Compute Class Weights
# Convert classes to a numpy array
classes = np.array([0, 1])
flat_train_labels = [label for sublist in train_labels for label in sublist]
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=flat_train_labels)
accelerator = Accelerator(mixed_precision="fp16" if HALF_PRECISION else None)
class_weights = torch.tensor(class_weights, dtype=torch.float16 if HALF_PRECISION else torch.float32).to(accelerator.device)

ssp_classes = np.array([0, 1, 2])
flat_ssp_labels = [label for sublist in train_ssp_labels for label in sublist]
ssp_class_weights = compute_class_weight(class_weight='balanced', classes=ssp_classes, y=flat_ssp_labels)
ssp_class_weights = torch.tensor(ssp_class_weights, dtype=torch.float16 if HALF_PRECISION else torch.float32).to(accelerator.device)


def compute_weights(training_labels):
  total_samples = sum(len(seq) for seq in training_labels)
  class_0_samples = sum(seq.count(0) for seq in training_labels)
  class_1_samples = sum(seq.count(1) for seq in training_labels)

  weight_0 = total_samples / (2 * class_0_samples)
  weight_1 = total_samples / (2 * class_1_samples)
  return weight_0, weight_1
weight_0, weight_1 = compute_weights(train_labels)
print(weight_0, weight_1)


def compute_weights(training_labels):
    total_samples = sum(len(seq) for seq in training_labels)
    class_counts = {
        'H': sum(seq.count('H') for seq in training_labels),
        'E': sum(seq.count('E') for seq in training_labels),
        'C': sum(seq.count('C') for seq in training_labels),
    }

    # Compute weights: inverse frequency normalized by the number of classes (3)
    weights = {cls: total_samples / (3 * count) for cls, count in class_counts.items() if count > 0}
    
    return weights['H'], weights['E'], weights['C']

# Example usage
weight_H, weight_E, weight_C = compute_weights(train_ssp)
print(weight_H, weight_E, weight_C)


train_masks = make_mask(train_sequences)
validation_masks = make_mask(validation_sequences)
test_masks = make_mask(test_sequences)

print("checking label lengths")
for i, (lbl, ssp_lbl) in enumerate(zip(train_labels, train_ssp_labels)):
#    print(len(lbl), len(ssp_lbl))
    if len(lbl) !=len(ssp_lbl):
        print(f"Mismatch at index {i}: labels has {len(lbl)}, ssp_labels has {len(ssp_lbl)}")



my_train = {"sequence" : train_sequences, "labels" : train_labels, "ssp_labels" : train_ssp_labels, "mask" : train_masks}
my_train = pd.DataFrame(my_train)
my_valid = {"sequence" : validation_sequences, "labels" : validation_labels, "ssp_labels" : validation_ssp_labels, "mask" : validation_masks}
my_valid = pd.DataFrame(my_valid)
my_test = {"sequence" : test_sequences, "labels" : test_labels, "ssp_labels" : test_ssp_labels, "mask" : test_masks}
my_test = pd.DataFrame(my_test)

my_train.head(5)

my_valid.head(5)

"""# PT5 Model and Low Rank Adaptation

## LoRA modification definition

Implementation taken from https://github.com/r-three/t-few

(https://github.com/r-three/t-few/blob/master/src/models/lora.py, https://github.com/r-three/t-few/tree/master/configs)
"""

config = {
    "lora_alpha": 1, #try 0.5, 1, 2, ..., 16
    "lora_dropout": 0.2,
    "lr": 5.701568055793089e-04,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 0.5,
    "num_train_epochs": 20,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "r": 2,
    "weight_decay": 0.2,
    # Add other hyperparameters as needed
}



# Modifies an existing transformer and introduce the LoRA layers

class LoRAConfig:
    def __init__(self):
        self.lora_rank = 4
        self.lora_init_scale = 0.01
        self.lora_modules = ".*SelfAttention|.*EncDecAttention"
        self.lora_layers = "q|k|v|o"
        self.trainable_param_names = ".*layer_norm.*|.*lora_[ab].*"
        self.lora_scaling_rank = 1
        # lora_modules and lora_layers are speicified with regular expressions
        # see https://www.w3schools.com/python/python_regex.asp for reference

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, scaling_rank, init_scale):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling_rank = scaling_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        if self.rank > 0:
            self.lora_a = nn.Parameter(torch.randn(rank, linear_layer.in_features) * init_scale)
            if init_scale < 0:
                self.lora_b = nn.Parameter(torch.randn(linear_layer.out_features, rank) * init_scale)
            else:
                self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        if self.scaling_rank:
            self.multi_lora_a = nn.Parameter(
                torch.ones(self.scaling_rank, linear_layer.in_features)
                + torch.randn(self.scaling_rank, linear_layer.in_features) * init_scale
            )
            if init_scale < 0:
                self.multi_lora_b = nn.Parameter(
                    torch.ones(linear_layer.out_features, self.scaling_rank)
                    + torch.randn(linear_layer.out_features, self.scaling_rank) * init_scale
                )
            else:
                self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features, self.scaling_rank))

    def forward(self, input):
        if self.scaling_rank == 1 and self.rank == 0:
            # parsimonious implementation for ia3 and lora scaling
            if self.multi_lora_a.requires_grad:
                hidden = F.linear((input * self.multi_lora_a.flatten()), self.weight, self.bias)
            else:
                hidden = F.linear(input, self.weight, self.bias)
            if self.multi_lora_b.requires_grad:
                hidden = hidden * self.multi_lora_b.flatten()
            return hidden
        else:
            # general implementation for lora (adding and scaling)
            weight = self.weight
            if self.scaling_rank:
                weight = weight * torch.matmul(self.multi_lora_b, self.multi_lora_a) / self.scaling_rank
            if self.rank:
                weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
            return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, rank={}, scaling_rank={}".format(
            self.in_features, self.out_features, self.bias is not None, self.rank, self.scaling_rank
        )


def modify_with_lora(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        LoRALinear(layer, config.lora_rank, config.lora_scaling_rank, config.lora_init_scale),
                    )
    return transformer

"""## Classification model definition

adding a token classification head on top of the encoder model

modified from [EsmForTokenClassification](https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/esm/modeling_esm.py#L1178)
"""

class ClassConfig:
    def __init__(self, dropout=0.2, num_labels=2, weight_0=1, weight_1=1):
        self.dropout_rate = dropout
        self.num_labels = num_labels
        print(weight_0, weight_1)
        self.weights=[weight_0, weight_1]


class ssConfig:
    def __init__(self, dropout=0.2, num_labels=3, weight_C=1, weight_H=1, weight_E=1):
        self.dropout_rate = dropout
        self.num_labels = num_labels
        print(weight_C, weight_H, weight_E)
        self.weights=[weight_C, weight_H, weight_E]



class T5EncoderForTokenClassification(T5PreTrainedModel):

    def __init__(self, config: T5Config, class_config, ss_config, half_precision=True):
        super().__init__(config)
        self.num_labels = class_config.num_labels
        self.num_ss_labels = ss_config.num_labels

        self.weights = torch.tensor(class_config.weights).cuda()
        self.ss_weights = torch.tensor(ss_config.weights).cuda()
        if half_precision:
            self.weights = self.weights.half()
            self.ss_weights = self.ss_weights.half()

        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(class_config.dropout_rate)
        #self.classifier = nn.Linear(config.hidden_size, class_config.num_labels)
        hidden_dim = int(config.hidden_size / 2)
        num_hidden_layers = 1 # Number of hidden layers in ConvBert.
        nlayers = 1 # Number of ConvBert layers.
        nhead = 4
        dropout = 0.2
        conv_kernel_size = 7

        self.classifier = ankh.ConvBertForMultiClassClassification(num_tokens=class_config.num_labels,
                                                                input_dim=config.hidden_size,
                                                                nhead=nhead,
                                                                hidden_dim=hidden_dim,
                                                                num_hidden_layers=num_hidden_layers,
                                                                num_layers=nlayers,
                                                                kernel_size=conv_kernel_size,
                                                                dropout=dropout)


        # **Secondary structure prediction head**
        #self.classifier_ss = ankh.ConvBertForMultiClassClassification(
        #    num_tokens=ss_config.num_labels,
        #    input_dim=config.hidden_size,
        #    nhead=nhead,
        #    hidden_dim=hidden_dim,
        #    num_hidden_layers=num_hidden_layers,
        #    num_layers=nlayers,
        #    kernel_size=conv_kernel_size,
        #    dropout=dropout
        #)
        self.classifier_ss = nn.Linear(config.hidden_size, ss_config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.classifier_ss = self.classifier_ss.to(self.encoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.classifer = self.classifier.to("cpu")
        self.classifier_ss = self.classifier_ss.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        ssp_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits_allosteric = self.classifier(sequence_output)
        logits_ss = self.classifier_ss(sequence_output)

        ssp_labels = torch.cat([ssp_labels, torch.tensor([[-100]], device=ssp_labels.device)], dim=1)

        #print("Last allosteric label:", labels[-1])
        #print("Last ssp label:", ssp_labels[-1])


        #print("checking label lengths")
        #for i, (lbl, ssp_lbl) in enumerate(zip(labels, ssp_labels)):
        #    print(len(lbl), len(ssp_lbl))
        #    if len(lbl) !=len(ssp_lbl):
        #        print(f"Mismatch at index {i}: labels has {len(lbl)}, ssp_labels has {len(ssp_lbl)}")

        loss = None
        loss_allo = None
        loss_ss= None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.weights)
            active_loss = attention_mask.view(-1) == 1
            #print("active loss shape:", active_loss.shape)
            #print("labels shape:", labels.shape)
            # Allosteric classification loss
            active_logits_allo = logits_allosteric.logits.view(-1, self.num_labels)
            #active_logits = outputs.logits.view(-1, self.num_labels)

            active_labels_allo = torch.where(
              active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
            )

            valid_logits_allo=active_logits_allo[active_labels_allo!=-100]
            valid_labels_allo=active_labels_allo[active_labels_allo!=-100]
            valid_labels_allo=valid_labels_allo.type(torch.LongTensor).to('cuda:0')

            loss_allo = loss_fct(valid_logits_allo, valid_labels_allo)
            #print("loss allo = ", loss_allo)
        if ssp_labels is not None: 
            loss_fct_ss = CrossEntropyLoss(weight=self.ss_weights)
            active_loss = attention_mask.view(-1)==1
            # Secondary structure prediction loss
            #active_logits_ss = logits_ss.logits.view(-1, self.num_ss_labels)
            active_logits_ss = logits_ss.view(-1, self.num_ss_labels)
            active_labels_ss = torch.where(
                active_loss, ssp_labels.view(-1), torch.tensor(-100).type_as(ssp_labels)
            )

            valid_logits_ss = active_logits_ss[active_labels_ss != -100]
            valid_labels_ss = active_labels_ss[active_labels_ss != -100]
            valid_labels_ss = valid_labels_ss.type(torch.LongTensor).to('cuda:0')
            
            loss_ss = loss_fct_ss(valid_logits_ss, valid_labels_ss)
            #print("loss ssp = ", loss_ss)
        if loss_allo is not None and loss_ss is not None: 
            # **Total loss (weighted sum)**
            loss = loss_allo + loss_ss  
        elif loss_allo is not None:
            loss= loss_allo
        elif loss_ss is not None:
            loss = loss_ss
        #print("total loss = ", loss)
        if not return_dict:
            logits = logits_allosteric
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits_allosteric,  # Return as dictionary
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


"""## Modified ProtT5 model
this creates a ProtT5 model with prediction head and LoRA modification
"""

def PT5_classification_model(num_labels, num_labels_ss, weight_0, weight_1, weight_C, weight_H, weight_E, half_precision=True):
    # Load PT5 and tokenizer
    # possible to load the half preciion model (thanks to @pawel-rezo for pointing that out)
    if not half_precision:
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    elif half_precision and torch.cuda.is_available() :
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16).to(torch.device('cuda'))
    else:
          raise ValueError('Half precision can be run on GPU only.')

    # Create new Classifier model with PT5 dimensions
    class_config=ClassConfig(num_labels=num_labels, weight_0=weight_0, weight_1=weight_1)
    ss_config=ssConfig(num_labels=num_labels_ss, weight_C=weight_C, weight_H=weight_H, weight_E=weight_E)
    class_model=T5EncoderForTokenClassification(model.config,class_config,ss_config)

    # Set encoder and embedding weights to checkpoint weights
    class_model.shared=model.shared
    class_model.encoder=model.encoder

    # Delete the checkpoint model
    model=class_model
    del class_model

    # Print number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("ProtT5_Classfier\nTrainable Parameter: "+ str(params))


    # Print trainable Parameter
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("ProtT5_LoRA_Classfier\nTrainable Parameter: "+ str(params) + "\n")

    return model, tokenizer

"""# Training Definition

## Training functions
"""

def compute_metrics(pred):
      predictions, labels = pred

      # Convert predictions and labels to tensors (if not already)
      predictions = torch.tensor(predictions.logits)
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
      print("AUC: ", auc)
      print("APS: ", aps)
      print("Precision: ", precision)
      print("Recall: ", recall)
      print("F1: ", f1)
      print("Accuracy: ", accuracy)
      return {
        'AUC': auc,
        'APS': aps,
        'Precision': precision,
        'Recall': recall,
        'f1': f1,
        'Accuracy': accuracy
      }

# Set random seeds for reproducibility of your trainings run
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)

# Dataset creation
def create_dataset(tokenizer,seqs,labels, ssp_labels):
    tokenized = tokenizer(seqs, max_length=1024, padding=False, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    # we need to cut of labels after 1023 positions for the data collator to add the correct padding (1023 + 1 special tokens)
    labels = [l[:1023] for l in labels]
    ssp_labels = [l[:1023] for l in ssp_labels]

    dataset = dataset.add_column("labels", labels)
    dataset = dataset.add_column("ssp_labels", ssp_labels)

    return dataset


# Main training fuction
def train_per_residue(
        train_df,         #training data
        valid_df,         #validation data
        test_df,
        num_labels= 2,    #number of classes
        num_labels_ss=3,

        # effective training batch size is batch * accum
        # we recommend an effective batch size of 8
        batch= 4,         #for training
        accum= 2,         #gradient accumulation

        val_batch = 16,   #batch size for evaluation
        epochs= 30,       #training epochs
        lr= 3e-4,         #recommended learning rate
        seed= 42,         #random seed
        mixed= True,     #enable mixed precision training
        gpu= 1,         #gpu selection (1 for first gpu)
        weight_0=1,
        weight_1=1
        ):

    # Set gpu device
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu-1)

    # Set all random seeds
    set_seeds(seed)

    # load model
    #model, tokenizer = PT5_classification_model(num_labels=num_labels, weight_0=weight_0, weight_1=weight_1)

    # load model
    model, tokenizer = PT5_classification_model(num_labels, num_labels_ss, weight_0, weight_1, weight_C, weight_H, weight_E, half_precision=mixed)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Preprocess inputs
    # Replace uncommon AAs with "X"
    train_df["sequence"]=train_df["sequence"].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
    valid_df["sequence"]=valid_df["sequence"].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
    test_df["sequence"]=test_df["sequence"].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)

    # Add spaces between each amino acid for PT5 to correctly use them
    train_df['sequence']=train_df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    valid_df['sequence']=valid_df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    test_df['sequence']=test_df.apply(lambda row : " ".join(row["sequence"]), axis = 1)

    # Create Datasets
    train_set = create_dataset(tokenizer, list(my_train['sequence']), list(my_train['labels']), list(my_train['ssp_labels']))
    valid_set = create_dataset(tokenizer, list(my_valid['sequence']), list(my_valid['labels']), list(my_valid['ssp_labels']))
    test_set = create_dataset(tokenizer, list(my_test['sequence']), list(my_test['labels']), list(my_test['ssp_labels']))

    for i, (lbl, ssp_lbl) in enumerate(zip(list(train_set['labels']), list(train_set['ssp_labels']))):
    #    print(len(lbl), len(ssp_lbl))
        if len(lbl) !=len(ssp_lbl):
            print(f"Mismatch at index {i}: labels has {len(lbl)}, ssp_labels has {len(ssp_lbl)}")

    print(train_set.column_names)  # Should include "labels" and "ssp_labels"
    print(train_set[0])  # Should print both labels separately

    # Prepare model, optimizer, and data loaders with accelerator
    #model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
     #   model, optimizer, train_set, valid_set
    #)



    # Initialize your custom LoRAConfig
    #lora_config = LoRAConfig()

    # Use the LoRAConfig values to configure the PEFT config
    #peft_config = LoraConfig(
    #    task_type=TaskType.TOKEN_CLS,
    #    inference_mode=False,
    #    r=config["r"],
    #    lora_alpha=config["lora_alpha"],
    #    target_modules=[module for module in lora_config.lora_modules.split('|')],  # Use the modules from LoraConfig
    #    lora_dropout=config["lora_dropout"],
    #    bias="none"
    #)

    # Add model modification lora
    peft_config = LoraConfig(
      task_type=TaskType.TOKEN_CLS,
      inference_mode=False,
      r=config["r"],
      lora_alpha=config["lora_alpha"],
      target_modules=["q", "k", "v", "o"], # also try "dense_h_to_4h" and "dense_4h_to_h"
      lora_dropout=config["lora_dropout"],
      bias="none" # or "all" or "lora_only"
    )
    model = get_peft_model(model, peft_config)
    #model = accelerator.prepare(model)
    #train_dataset = accelerator.prepare(train_set)
    #test_dataset = accelerator.prepare(test_dataset)
    #valid_dataset = accelerator.prepare(valid_set)
    train_dataset = train_set
    valid_dataset = valid_set
    test_dataset = test_set

    # Huggingface Trainer arguments
    args = TrainingArguments(
        output_dir = "PT5_Perf_acc_ankh_head/",
        eval_strategy = "epoch",
        eval_steps = 500,
        logging_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=lr,
        lr_scheduler_type="cosine_with_restarts",
        per_device_train_batch_size=batch,
        #per_device_eval_batch_size=val_batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        weight_decay = 0.2,
        seed = seed,
        fp16 = mixed,
        remove_unused_columns=False,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        logging_dir="./logs",  # Ensure this is set
        logging_first_step=True,  # Ensure logging happens on the first step
        logging_steps=50,  # Adjust for your dataset size.
        report_to="wandb",
        push_to_hub=False,
        no_cuda=False,
        greater_is_better=True,
        save_total_limit=7,

    )

    # For token classification we need a data collator here to pad correctly
    data_collator = DataCollatorForTokenClassification(tokenizer)
    print(f"Evaluation dataset size: {len(valid_set)}")

    # Trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,

    )
    # Manually trigger evaluation
    #eval_results = trainer.evaluate(valid_set)
    # Train model
    trainer.train()
    # Manually trigger evaluation
    #eval_results = trainer.evaluate(valid_set)
    #print(eval_results)
    eval_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("Evaluation Metrics:", eval_metrics)

    return tokenizer, model, trainer.state.log_history

"""# Run Training

## Training
"""
print(my_train.iloc[:3])
print(f"Evaluation dataset size: {len(my_valid)}")

tokenizer, model, history = train_per_residue(my_train, my_valid, my_test, num_labels=2, num_labels_ss=3, batch=1, accum=1, epochs=40, seed=42, gpu=2, weight_0=weight_0, weight_1=weight_1)


