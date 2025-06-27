
"""# Prot T5 Finetuning
# per residue classification

This notebook allows you to finetune PLMs to your own datasets

For better perfomance we apply [Parameter-Efficient Fine-Tuning (PEFT)](https://huggingface.co/blog/peft). For this we apply [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

The core training loop is implemented with the pytorch [huggingface trainer](https://huggingface.co/docs/transformers/main_classes/trainer).

In case it is needed for higher memory efficiency, we utilize the [deepspeed](https://github.com/microsoft/DeepSpeed) implementation of [huggingface](https://huggingface.co/docs/accelerate/usage_guides/deepspeed).

## Imports and env. variables
"""
import os
import os.path
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#import dependencies
#os.chdir("set working path here")

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
from transformers import T5EncoderModel, T5Tokenizer
from transformers.models.esm.modeling_esm import EsmPreTrainedModel, EsmModel
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, set_seed
from transformers import DataCollatorForTokenClassification
from transformers import AutoConfig

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# for custom DataCollator
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

import peft
from peft import get_peft_config, PeftModel, PeftConfig, inject_adapter_in_model, LoraConfig, TaskType

from evaluate import load
from datasets import Dataset

from tqdm import tqdm
import random

from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    f1_score,
    average_precision_score,
    recall_score,
    precision_score,
    matthews_corrcoef
)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ast
from sklearn.utils.class_weight import compute_class_weight
import ankh
# Clear GPU cache
torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

"""**For easy setup of this environment you can use the finetuning.yml File provided in this folder**

check here for [setting up env from a yml File](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

# Valid Model checkpoints

This notebook was tested with all models mentioned below.
All required, model specific adaptations will be taken care of.
"""

ESMs = ["facebook/esm2_t6_8M_UR50D",
         "facebook/esm2_t12_35M_UR50D",
         "facebook/esm2_t30_150M_UR50D",
         "facebook/esm2_t33_650M_UR50D",
         "facebook/esm2_t36_3B_UR50D"]

T5s = ["Rostlab/prot_t5_xl_uniref50",
       'Rostlab/ProstT5',
       "ElnaggarLab/ankh-base",
       "ElnaggarLab/ankh-large"]

"""### Select your model:"""

checkpoint = T5s[1]
"""# Input data

Provide your training and validation data in seperate pandas dataframes

example shown below

**Modify the data loading part above as needed for your data**

To run the training you need two dataframes (training and validation) each with the columns "sequence" and "label" and "mask"

Columns are:
+ protein sequence
+ label is a list of len(protein sequence) with integers (from 0 to number of classes - 1) corresponding to predicted class at this position
+ mask gives the possibility to ignore parts of the positions. Provide a list of len(protein sequence) where 1 is processed, while 0 is ignored
"""
head = 'ConvBert'
train_file = '~/PLM_Allosteric_Classification/data/train_df.csv'
test_file = '~/PLM_Allosteric_Classification/data/test_df.csv'
train_df = pd.read_csv(train_file)
test_df_total = pd.read_csv(test_file)
train_sequences = train_df['Sequences'].tolist()
train_labels = train_df['Labels'].apply(ast.literal_eval).tolist()
test_total_sequences = test_df_total['Sequences'].tolist()
test_total_labels = test_df_total['Labels'].apply(ast.literal_eval).tolist()

# Further split temp into validation and test (e.g., 50% of 30% -> 15% each)
validation_sequences, test_sequences, validation_labels, test_labels = train_test_split(test_total_sequences, test_total_labels, test_size=0.5, random_state=42)

num_labels = 2
classes = np.array([0, 1])
flat_train_labels = [label for sublist in train_labels for label in sublist]
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=flat_train_labels)

def make_mask(sequences):
  masks = []
  for seq in sequences:
    masks.append([1]*len(seq))
  return masks

train_masks = make_mask(train_sequences)
validation_masks = make_mask(validation_sequences)
test_masks = make_mask(test_sequences)

my_train = {"sequence" : train_sequences, "label" : train_labels, "mask" : train_masks}
my_train = pd.DataFrame(my_train)
my_valid = {"sequence" : validation_sequences, "label" : validation_labels, "mask" : validation_masks}
my_valid = pd.DataFrame(my_valid)
my_test = {"sequence" : test_sequences, "label" : test_labels, "mask" : test_masks}
my_test = pd.DataFrame(my_test)

my_train.head(5)

my_valid.head(5)

"""# Models and Low Rank Adaptation

## T5 Models

### Classification model definition

adding a token classification head on top of the encoder model

modified from [EsmForTokenClassification](https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/esm/modeling_esm.py#L1178)
"""


class TokenFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=5.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Tensor of shape [num_classes] or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [batch_size, seq_len, num_classes]
        targets: [batch_size, seq_len]  (values in [0, num_classes-1])
        """

        # Flatten the input
        logits = logits.view(-1, logits.size(-1))     # [batch*seq_len, num_classes]
        targets = targets.view(-1)                    # [batch*seq_len]
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [batch*seq_len]

        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            # Get alpha for each target class
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # No reduction


class ClassConfig:
    def __init__(self, dropout=0.2, num_labels=2, class_weights=[1,1], head = 'LogisticRegression', loss_type="CELoss"):
        self.dropout_rate = dropout
        self.num_labels = num_labels
        self.class_weights = class_weights
        self.head = head
        self.loss_type = loss_type

class T5EncoderForTokenClassification(T5PreTrainedModel):

    def __init__(self, config: T5Config, class_config):
        super().__init__(config)
        self.num_labels = class_config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self.class_weights = class_config.class_weights
        self.loss_type = class_config.loss_type
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(class_config.dropout_rate)
        if head == 'LogisticRegression':
            self.classifier = nn.Linear(config.hidden_size, class_config.num_labels)
        elif head == 'ConvBert':
            hidden_size = config.hidden_size
            hidden_dim = int(hidden_size / 2)
            num_hidden_layers = 1 # Number of hidden layers in ConvBert.
            nlayers = 1 # Number of ConvBert layers.
            nhead = 4
            dropout = 0.2
            conv_kernel_size = 7
            self.classifier = ankh.ConvBertForMultiClassClassification(num_tokens=self.num_labels,
                                                                    input_dim=hidden_size,
                                                                    nhead=nhead,
                                                                    hidden_dim=hidden_dim,
                                                                    num_hidden_layers=num_hidden_layers,
                                                                    num_layers=nlayers,
                                                                    kernel_size=conv_kernel_size,
                                                                    dropout=dropout)


        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.head = class_config.head

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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
        logits = self.classifier(sequence_output)

        if isinstance(logits, TokenClassifierOutput):
            logits = logits.logits  # Extract the tensor


        loss = None
        if labels is not None:
            
            if self.loss_type == "CELoss":
                loss_fct = CrossEntropyLoss(weight = self.class_weights, ignore_index=-100) 
            elif self.loss_type == "Focal":
                loss_fct = TokenFocalLoss(alpha=self.class_weights, gamma=1.0)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)

            active_labels = torch.where(
              active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
            )

            valid_logits=active_logits[active_labels!=-100]
            valid_labels=active_labels[active_labels!=-100]

            valid_labels=valid_labels.type(torch.LongTensor).to('cuda:0')

            loss = loss_fct(valid_logits, valid_labels)


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

"""### Load T5 model
this creates a T5 model with prediction head and LoRA modification
"""

def load_T5_model_classification(checkpoint, num_labels, class_weights, half_precision, full = False, deepspeed=True, head = 'LogisticRegression', loss_type = "CELoss"):
    # Load model and tokenizer

    if "ankh" in checkpoint :
        model = T5EncoderModel.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    elif "prot_t5" in checkpoint:
        # possible to load the half precision model (thanks to @pawel-rezo for pointing that out)
        if half_precision and deepspeed :
            tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
            model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16)#.to(torch.device('cuda')
        else:
            model = T5EncoderModel.from_pretrained(checkpoint)
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)

    elif "ProstT5" in checkpoint:
        if half_precision and deepspeed:
            tokenizer = T5Tokenizer.from_pretrained(checkpoint, do_lower_case=False)
            model = T5EncoderModel.from_pretrained(checkpoint, torch_dtype=torch.float16)#.to(torch.device('cuda')
        else:
            model = T5EncoderModel.from_pretrained(checkpoint)
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)

    # Create new Classifier model with PT5 dimensions
    class_config=ClassConfig(num_labels=num_labels, class_weights=class_weights, head=head, loss_type=loss_type)
    class_model=T5EncoderForTokenClassification(model.config,class_config)

    # Set encoder and embedding weights to checkpoint weights
    class_model.shared=model.shared
    class_model.encoder=model.encoder

    # Delete the checkpoint model
    model=class_model
    del class_model

    if full == True:
        return model, tokenizer

    # Print number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("T5_Classfier\nTrainable Parameter: "+ str(params))


    # lora modification
    peft_config = LoraConfig(
        r=8, lora_alpha=16, bias="none", target_modules=["q","k","v"], task_type=TaskType.TOKEN_CLS, inference_mode=False, lora_dropout=0.1,
    )


    model = inject_adapter_in_model(peft_config, model)

    # Unfreeze the prediction head
    for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True

    # Print trainable Parameter
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("T5_LoRA_Classfier\nTrainable Parameter: "+ str(params) + "\n")

    return model, tokenizer

"""## ESM2 Models

### Classification model definition and DataCollator
"""

class EsmForTokenClassificationCustom(EsmPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.class_weights = config.class_weights
        self.loss_type = config.loss_type

        if self.class_weights is not None and not isinstance(self.class_weights, torch.Tensor):
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32).to(device)  # put on same device as model


        self.esm = EsmModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        head = config.head
        if head == 'LogisticRegression':
            self.classifier = nn.Linear(config.hidden_size, class_config.num_labels)
        elif head == 'ConvBert':
            hidden_size = config.hidden_size
            hidden_dim = int(hidden_size / 2)
            num_hidden_layers = 1 # Number of hidden layers in ConvBert.
            nlayers = 1 # Number of ConvBert layers.
            nhead = 4
            dropout = 0.2
            conv_kernel_size = 7
            self.classifier = ankh.ConvBertForMultiClassClassification(num_tokens=self.num_labels,
                                                                    input_dim=hidden_size,
                                                                    nhead=nhead,
                                                                    hidden_dim=hidden_dim,
                                                                    num_hidden_layers=num_hidden_layers,
                                                                    num_layers=nlayers,
                                                                    kernel_size=conv_kernel_size,
                                                                    dropout=dropout)



        self.init_weights()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if isinstance(logits, TokenClassifierOutput):
            logits = logits.logits
        loss = None
        # changed to ignore special tokens at the seq start and end
        # as well as invalid positions (labels -100)
        if labels is not None:

            if self.loss_type == "CELoss":
                loss_fct = CrossEntropyLoss(weight = self.class_weights, ignore_index=-100)
            elif self.loss_type == "Focal":
                loss_fct = TokenFocalLoss(alpha=self.class_weights, gamma=2.0)

            loss_fct = CrossEntropyLoss(weight = self.class_weights, ignore_index=-100) if self.class_weights is not None else CrossEntropyLoss(ignore_index=-100)

            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)

            active_labels = torch.where(
              active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
            )

            valid_logits=active_logits[active_labels!=-100]
            valid_labels=active_labels[active_labels!=-100]

            valid_labels=valid_labels.type(torch.LongTensor).to('cuda:0')

            loss = loss_fct(valid_logits, valid_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# based on transformers DataCollatorForTokenClassification
@dataclass
class DataCollatorForTokenClassificationESM(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                # to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
                # changed to pad the special tokens at the beginning and end of the sequence
                [self.label_pad_token_id] + to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)-1) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.float)
        return batch

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()

"""### Load ESM2 Model"""

#load ESM2 models
def load_esm_model_classification(checkpoint, num_labels, class_weights, half_precision, full=False, deepspeed=True, head = 'LogisticRegression', loss_type = "CELoss"):

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    config = AutoConfig.from_pretrained(checkpoint)
    config.num_labels = num_labels
    config.class_weights = class_weights.tolist() if isinstance(class_weights, torch.Tensor) else class_weights
    config.head = head
    config.loss_type = loss_type
    if half_precision and deepspeed:
        model = EsmForTokenClassificationCustom.from_pretrained(checkpoint, config = config, torch_dtype = torch.float16)
    else:
        model = EsmForTokenClassificationCustom.from_pretrained(checkpoint, config = config)

    if full == True:
        return model, tokenizer

    #peft_config = LoraConfig(
    #    r=8, lora_alpha=16, bias="all", target_modules=["query","key","value","dense"]
    #)


    # lora modification
    peft_config = LoraConfig(
        r=8, lora_alpha=16, bias="none", target_modules=["query","key","value"], task_type=TaskType.TOKEN_CLS, inference_mode=False,
    )



    model = inject_adapter_in_model(peft_config, model)

    # Unfreeze the prediction head
    for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True

    return model, tokenizer

"""# Training Definition

## Deepspeed config
"""

# Deepspeed config for optimizer CPU offload

ds_config = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}

"""## Training functions"""

# Set random seeds for reproducibility of your trainings run
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)

# Dataset creation
def create_dataset(tokenizer,seqs,labels,checkpoint):

    tokenized = tokenizer(seqs, max_length=1024, padding=True, truncation=True)
    dataset = Dataset.from_dict(tokenized)

    if ("esm" in checkpoint) or ("ProstT5" in checkpoint):
        # we need to cut of labels after 1022 positions for the data collator to add the correct padding (1022 + 2 special tokens)
        labels = [l[:1022] for l in labels]
    else:
        # we need to cut of labels after 1023 positions for the data collator to add the correct padding (1023 + 1 special tokens)
        labels = [l[:1023] for l in labels]

    dataset = dataset.add_column("labels", labels)

    return dataset

# Main training fuction
def train_per_residue(
        checkpoint,       #model checkpoint

        train_df,         #training data
        valid_df,         #validation data
        test_df,          #test data
        num_labels = 2,   #number of classes

        # effective training batch size is batch * accum
        # we recommend an effective batch size of 8
        batch = 4,        #for training
        accum = 2,        #gradient accumulation

        val_batch = 16,   #batch size for evaluation
        epochs = 10,      #training epochs
        lr = 3e-4,        #recommended learning rate
        seed = 42,        #random seed
        deepspeed = False,#if gpu is large enough disable deepspeed for training speedup
        mixed = True,     #enable mixed precision training
        full = False,     #enable training of the full model (instead of LoRA)
        gpu = 1,           #gpu selection (1 for first gpu)
        class_weights = None,
        head = 'ConvBert',
        loss_type = "CEloss"
        ):

    print("Model used:", checkpoint, "\n")

    # Correct incompatible training settings
    if "ankh" in checkpoint and mixed:
        print("Ankh models do not support mixed precision training!")
        print("switched to FULL PRECISION TRAINING instead")
        mixed = False

    class_weights = torch.tensor(class_weights, dtype=torch.float16 if mixed else torch.float32).to(device)


    # Set gpu device
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu-1)

    # Set all random seeds
    set_seeds(seed)

    # load model
    if "esm" in checkpoint:
        model, tokenizer = load_esm_model_classification(checkpoint, num_labels, class_weights,  mixed, full, deepspeed, head, loss_type)
    else:
        model, tokenizer = load_T5_model_classification(checkpoint, num_labels, class_weights, mixed, full, deepspeed, head, loss_type)

    # Preprocess inputs
    # Replace uncommon AAs with "X"
    train_df["sequence"]=train_df["sequence"].str.replace('|'.join(["O","B","U","Z","J"]),"X",regex=True)
    valid_df["sequence"]=valid_df["sequence"].str.replace('|'.join(["O","B","U","Z","J"]),"X",regex=True)
    test_df["sequence"]=test_df["sequence"].str.replace('|'.join(["O","B","U","Z","J"]),"X",regex=True)

    # Add spaces between each amino acid for ProtT5 and ProstT5 to correctly use them
    if "Rostlab" in checkpoint:
        train_df['sequence']=train_df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
        valid_df['sequence']=valid_df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
        test_df['sequence']=test_df.apply(lambda row : " ".join(row["sequence"]), axis = 1)

    # Add <AA2fold> for ProstT5 to inform the model of the input type (amino acid sequence here)
    if "ProstT5" in checkpoint:
        train_df['sequence']=train_df.apply(lambda row : "<AA2fold> " + row["sequence"], axis = 1)
        valid_df['sequence']=valid_df.apply(lambda row : "<AA2fold> " + row["sequence"], axis = 1)
        test_df['sequence']=test_df.apply(lambda row : "<AA2fold> " + row["sequence"], axis = 1)

    # Create Datasets
    train_set=create_dataset(tokenizer,list(train_df['sequence']),list(train_df['label']),checkpoint)
    valid_set=create_dataset(tokenizer,list(valid_df['sequence']),list(valid_df['label']),checkpoint)
    test_set=create_dataset(tokenizer,list(test_df['sequence']),list(test_df['label']),checkpoint)


    # Huggingface Trainer arguments
    args = TrainingArguments(
        "./scripts/Finetuning/PT5/",
        evaluation_strategy = "epoch", #we set this to "steps" because we train only for a single epoch here
        eval_steps = 500,              #but still want to get intermediate evaluation results
        logging_strategy = "epoch",
        save_strategy = "no",
        learning_rate=lr,
        lr_scheduler_type= "cosine",
        per_device_train_batch_size=batch,
        #per_device_eval_batch_size=val_batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed = seed,
        deepspeed= ds_config if deepspeed else None,
        fp16 = mixed,
        metric_for_best_model="APS",
        weight_decay = 0.2,

    )


    #best_threshold = None
    # Metric definition for validation data
    def compute_metrics(eval_pred):
        #global best_threshold 
        metric = load("accuracy")
        predictions, labels = eval_pred # logits and true labels

        # Convert predictions and labels to tensors (if not already)
        predictions = torch.tensor(predictions)
        labels = torch.tensor(labels)

        # Flatten predictions and labels
        predictions = predictions.view(-1, predictions.shape[-1])  # (N_tokens, n_classes)
        labels = labels.view(-1)  # (N_tokens,)

        # Create mask for valid tokens
        mask = labels != -100

        # calculate softmax probabilities
        probs = F.softmax(predictions[mask].float(), dim=-1)

        preds = torch.argmax(probs, dim=-1).reshape(-1)

        labels = labels[mask]

        # Convert to numpy for sklearn
        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy() 


        pos_probs = probs_np[:, 1]


        accuracy = accuracy_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_np, average="binary", pos_label=1)
        precision = precision_score(labels_np, preds_np, average="binary", pos_label=1)
        recall = recall_score(labels_np, preds_np, average="binary", pos_label=1)
        auc = roc_auc_score(labels_np, pos_probs)
        aps = average_precision_score(labels_np, pos_probs)
        print("AUC:  ", auc)
        print("APS:  ", aps)
        print("Precision:  ", precision)
        print("Recall:   ", recall)
        print("F1:  ", f1)
        print("Accuracy:  ", accuracy)
        return {
            'AUC': auc,
            'APS': aps,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy,
        }

        #return metric.compute(predictions=predictions, references=labels)

    # For token classification we need a data collator here to pad correctly
    # For esm2 and Prost pad at the beginning and at the end
    if ("esm" in checkpoint) or ("ProstT5" in checkpoint):
        data_collator = DataCollatorForTokenClassificationESM(tokenizer)
    # For Ankh and ProtT5 pad only at the end
    else:
        data_collator = DataCollatorForTokenClassification(tokenizer)

    # Trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()


    test_metrics = trainer.evaluate(eval_dataset=test_set)
    print("Test Metrics:", test_metrics)


    return tokenizer, model, trainer.state.log_history

"""# Run Training

## Training
"""

tokenizer, model, history = train_per_residue(checkpoint, my_train, my_valid, my_test, num_labels=2, batch=2, accum=8, epochs=30, seed=42, gpu=2, mixed = True, class_weights=class_weights, head = head, loss_type = "Focal")


"""# Save and Load the finetuned model"""

def save_model(model,filepath):
# Saves all parameters that were changed during finetuning

    # Create a dictionary to hold the non-frozen parameters
    non_frozen_params = {}

    # Iterate through all the model parameters
    for param_name, param in model.named_parameters():
        # If the parameter has requires_grad=True, add it to the dictionary
        if param.requires_grad:
            non_frozen_params[param_name] = param

    # Save only the finetuned parameters
    torch.save(non_frozen_params, filepath)


def load_model(checkpoint, filepath, num_labels=2, class_weights = class_weights, mixed = True, full = False, deepspeed=True):
# Creates a new PT5 model and loads the finetuned weights from a file

    # load model
    if "esm" in checkpoint:
        model, tokenizer = load_esm_model_classification(checkpoint, num_labels, class_weights, mixed, full, deepspeed)
    else:
        model, tokenizer = load_T5_model_classification(checkpoint, num_labels, class_weights, mixed, full, deepspeed)

    # Load the non-frozen parameters from the saved file
    non_frozen_params = torch.load(filepath)

    # Assign the non-frozen parameters to the corresponding parameters of the model
    for param_name, param in model.named_parameters():
        if param_name in non_frozen_params:
            param.data = non_frozen_params[param_name].data

    return tokenizer, model

"""This saves only the finetuned weights to a .pth file

The file has a size of only a few MB, while the entire model would be around 4.8 GB
"""

if ("esm" in checkpoint): 
    model_name = "ESM" 
if ("prot_t5_xl" in checkpoint): 
    model_name = "ProtT5" 
if ("ankh-large" in checkpoint):
    model_name = "Ankh Large" 


save_path = f"./{model_name}_secstr_finetuned.pth"
save_model(model, save_path)

"""To load the weights again, we initialize a new PT5 model from the pretrained checkpoint and load the LoRA weights afterwards

You need to specifiy the correct num_labels here
"""

tokenizer, model_reload = load_model(checkpoint, save_path, num_labels=num_labels)

"""To check if the original and the reloaded models are identical we can compare weights"""

# Put both models to the same device
model=model.to("cpu")
model_reload=model_reload.to("cpu")

# Iterate through the parameters of the two models and compare the data
for param1, param2 in zip(model.parameters(), model_reload.parameters()):
    if not torch.equal(param1.data, param2.data):
        print("Models have different weights")
        break
else:
    print("Models have identical weights")

"""# Make predictions on a test set

This time we take the test data we prepared before
"""

# Drop unneeded columns (remember, mask was already included as -100 values to label)
my_test=my_test[["sequence","label"]]

# Preprocess sequences
my_test["sequence"]=my_test["sequence"].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
my_test['sequence']=my_test.apply(lambda row : " ".join(row["sequence"]), axis = 1)
my_test.head(5)

"""Then we create predictions on our test data using the model we trained before"""

# Set the device to use
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Create Dataset
test_set=create_dataset(tokenizer,list(my_test['sequence']),list(my_test['label']),checkpoint)
# Make compatible with torch DataLoader
test_set = test_set.with_format("torch", device=device)

# For token classification we need a data collator here to pad correctly
if ("esm" in checkpoint) or ("ProstT5" in checkpoint):
    data_collator = DataCollatorForTokenClassificationESM(tokenizer)
# For Ankh and ProtT5 pad only at the end
else:
    data_collator = DataCollatorForTokenClassification(tokenizer)

# Create a dataloader for the test dataset
test_dataloader = DataLoader(test_set, batch_size=16, shuffle = False, collate_fn = data_collator)

# Put the model in evaluation mode
model.eval()

# Make predictions on the test dataset
predictions = []
# We need to collect the batch["labels"] as well, this allows us to filter out all positions with a -100 afterwards
padded_labels = []

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # Padded labels from the data collator
        padded_labels += batch['labels'].tolist()
        # Add batch results(logits) to predictions, we take the argmax here to get the predicted class
        predictions += model.float()(input_ids, attention_mask=attention_mask).logits.argmax(dim=-1).tolist()

"""Finally, we compute our desired performance metric for the test data"""

# to make it easier we flatten both the label and prediction lists
def flatten(l):
    return [item for sublist in l for item in sublist]

# flatten and convert to np array for easy slicing in the next step
predictions = np.array(flatten(predictions))
padded_labels = np.array(flatten(padded_labels))

# Filter out all invalid (label = -100) values
predictions = predictions[padded_labels!=-100]
padded_labels = padded_labels[padded_labels!=-100]

# Calculate classification Accuracy
print("Accuracy: ", accuracy_score(padded_labels, predictions))

"""Great, 84.6% Accuracy is a decent test performance for the "new_pisces" dataset (see results in [Table 7](https://ieeexplore.ieee.org/ielx7/34/9893033/9477085/supp1-3095381.pdf?arnumber=9477085) "NEW364" )

Further hyperparameter optimization will most likely increase performance
"""
