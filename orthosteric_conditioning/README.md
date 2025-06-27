# Orthosteric Conditioning for Protein Residue Classification
This folder contains a fine-tuning pipeline for per-residue classification of protein sequences, adapted to support orthosteric site conditioning. The core code is based on earlier work from (1) and the original code can be found in the [RSchmirler repo](https://github.com/RSchmirler/data-repo_plm-finetune-eval/tree/main), and has been significantly extended for our objectives.

## Overview
This script fine-tunes protein language models (including Ankh-large, ESM-2 3B and ProtT5 XL) to classify residues as part of an allosteric binding pocket, conditioning predictions on orthosteric embeddings. This approach, which captures functional dependencies between orthosteric and allosteric binding sites, improves upon embedding extraction and fine-tuning alone and achieves results comparable with leading sturcture-based allosteric site predictors.

Key features include:

✅ Pluggable classification head: Choose between a ConvBertForTokenClassificationHead head (context-aware)or a simple logistic regression layer.

✅ Custom loss functions: Use standard weighted cross-entropy or Focal Loss to address class imbalance.

✅ Orthosteric embedding conditioning: Incorporate orthosteric site information by augmenting the model’s input embeddings with fixed orthosteric site embeddings.


## Features
### Classification Heads
convbert: Applies a ConvBERT-style classification head for richer context modeling.

logistic: Uses a simple linear classifier over token embeddings.

### Loss Functions
CELoss: Weighted cross-entropy loss, configurable with class weights.

FocalLoss: Down-weights easy negatives and focuses training on difficult examples.

## Orthosteric Embedding Conditioning
You can provide orthosteric labels for each sequence, which are integrated into the input representations during fine-tuning. This enables conditional modeling of allosteric sites with respect to the orthosteric context.


## Acknowledgements
This codebase is adapted from [original group's repo or paper, if public], with substantial additions for conditional modeling and flexible experimentation.


## References
(1) Schmirler, R., Heinzinger, M. & Rost, B. Fine-tuning protein language models boosts predictions across diverse tasks. Nat Commun 15, 7407 (2024). https://doi.org/10.1038/s41467-024-51844-2
