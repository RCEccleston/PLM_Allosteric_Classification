# Embedding extraction and classification
This folder contains a pipeline for token-level classification of protein sequences, by extracting pLM embeddings to train a classifier to predict allosteric binding sites.

## Overview
This folder contains scripts for 3 protein language models (including Ankh-large, ESM-2 3B and ProtT5 XL) to classify residues as part of an allosteric binding pocket, using the ConvBertForTokenClassification by Ankh, the source code for which can be found [here](https://github.com/agemagician/Ankh/blob/main/src/ankh/models/convbert_multiclass_classification.py)

The '''TransferLearning''' folder contains scripts to pre-train the classifier on data from PDBbind, before training on the ASD data to predict allosteric sites.
