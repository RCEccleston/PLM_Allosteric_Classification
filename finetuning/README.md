# Fine-tuning pLMs for Protein Residue Classification
This folder contains a fine-tuning pipeline for token-level classification of protein sequences of allosteric binding sites. The core code is based on earlier work from (1) and the original code can be found in the [RSchmirler repo](https://github.com/RSchmirler/data-repo_plm-finetune-eval/tree/main), and has been significantly extended for our objectives.

## Overview
This script fine-tunes protein language models (including Ankh-large, ESM-2 3B and ProtT5 XL) to classify residues as part of an allosteric binding pocket.

Key features include:

✅ Pluggable classification head: Choose between a ConvBertForTokenClassificationHead head (context-aware)or a simple logistic regression layer.

✅ Custom loss functions: Use standard weighted cross-entropy or Focal Loss to address class imbalance.


## Features
### Classification Heads
convbert: Applies a ConvBERT-style classification head for richer context modeling.

logistic: Uses a simple linear classifier over token embeddings.

### Loss Functions
CELoss: Weighted cross-entropy loss, configurable with class weights.

FocalLoss: Down-weights easy negatives and focuses training on difficult examples.

## Acknowledgements
This codebase is adapted from [original group's repo or paper, if public], with substantial additions for conditional modeling and flexible experimentation.


## References
(1) Schmirler, R., Heinzinger, M. & Rost, B. Fine-tuning protein language models boosts predictions across diverse tasks. Nat Commun 15, 7407 (2024). https://doi.org/10.1038/s41467-024-51844-2
