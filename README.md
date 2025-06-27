# Predicting Allosteric Binding Sites using Protein Language Models
## Overview
This project explores the use of protein language models (pLMs) for token-level classification of allosteric binding sites. It includes three main methods:

* Embedding Extraction: Extracting token-level embeddings as input for a classification head.
* Fine-tuning: Adapting pLMs using LoRA, DeepSpeed, and PEFT to finetune them to the allosteric binding site dataset
* Orthosteric conditioning: A structure-aware conditioning mechanism in which orthosteric sites are encoded and integrated directly into the input embeddings


The models used in this study include:
* [Ankh-large](https://github.com/agemagician/Ankh)
* [ESM2](https://github.com/facebookresearch/esm)
* [Prot t5 xl](https://github.com/agemagician/ProtTrans/tree/master)

This project uses the classifier '''ConvBertForTokenClassification''', provided by the Ankh repository, the source code for which can be found [here](https://github.com/agemagician/Ankh/blob/main/src/ankh/models/convbert_multiclass_classification.py) 
## Datasets
There are 3 main datasets used in this project:

* Allosteric site database: ASD_Release_202306_XF which can be found [here](https://mdl.shsmu.edu.cn/ASD/) 
* PDBbind dataset: PDBbind v2020 which can be found [here](https://www.pdbbind-plus.org.cn/)
* Orthosteric-allosteric site subset from ASD database, which can be found [here](https://mdl.shsmu.edu.cn/ASD/) 

## Classification head
The classification head for the allosteric binding site classification task used in this project is a Convolutional Transformer-based Token Classifier, specifically designed for multiclass token classification by the team behind Ankh. 

## Project Structure
```bash
PLM_Allosteric_Classification/
├── data/ # folder containing data used in this project 
├── data_processing/ # folder of scripts to extract and process data
├── embedding_extraction/ # scripts to perform embedding extraction to train classifier
├── finetuning/ # script to fine-tune pLMs
├──  orthosteric_conditioning/ # script to fine-tune pLMs with orthosteric conditioning
├──  README.md         # Project documentation
```

# Notes on Model Usage

This repository does not contain the actual PLMs (Ankh, ESM2, ProtT5) due to their size and licensing. Instead, the scripts dynamically load them from Hugging Face or other sources. Ensure you have the appropriate access and permissions to use these models.


