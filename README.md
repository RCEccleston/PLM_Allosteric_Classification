# Predicting Allosteric Binding Sites using Protein Language Models
## Overview
This project explores the use of protein language models (PLMs) for token-level classification of allosteric binding sites. It includes two main tasks:

    * Embedding Extraction: Extracting token-level embeddings as input for a classification head.
    * Fine-tuning: Adapting PLMs using LoRA, DeepSpeed, and PEFT to finetune them to the allosteric binding site dataset.

The project suffers from issues due to lack of allosteric binding data and the sparseness of positive (ie allosteric) labels in the dataset. In an attempt to mitigate this the following techniques have been used:

    * Weighted cross-entropy loss
    * Transfer learning with larger PDBbind dataset
    * Contrastive loss
    * Focal loss
    * Segmented data
    * Multi-task learning with 2 classification heads: allosteric site prediction and secondard structure prediction 

The models used in this study include:

    * [Ankh-large](https://github.com/agemagician/Ankh) 
    * [ESM2](https://github.com/facebookresearch/esm)
    * [Prot t5 xl](https://github.com/agemagician/ProtTrans/tree/master)

## Datasets
There are 2 main datasets used in this project:

    * Allosteric site database: ASD_Release_202306_XF which can be found [here](https://mdl.shsmu.edu.cn/ASD/)
    
    * PDBbind dataset: PDBbind v2020 which can be found [here](https://www.pdbbind-plus.org.cn/)

The secondary structure prediction data used in the multi-task learning was generated using the Biopython DSSP module

## Project Structure
```bash
PLM_Allosteric_Classification/
├──  finetuning/
│   ├── Ankh/
|       ├── Ankh_large_finetuning.py # Script to perform finetuning of Ankh_large with classification head with weighted CE loss on ASD data
│   ├── ESM2/        
|       ├── esm2_t33_650m_ur50d_finetuning # Script to perform finetuning of esm2_t33_650m with classificatio head and weighted CE loss on ASD data
│   ├── ProtT5  
|       ├── ContrastiveLoss/
|           ├── pt5_lora_finetuning_per_residue_class_contrastive.py # Script to perform finetuning of PT5 using contrastive loss
│       ├── FocalLoss/
|           ├── pt5_lora_finetuning_per_residue_class_FL.py # Script to perform finetuning of PT5 using focal loss
|       ├── Segmented/
|           ├── pt5_lora_finetuning_per_residue_class_segmented.py # Script to perform finetuning of PT5 using segmented data technique 
│       ├── TransferLearning/           
|           ├── pt5_lora_finetuning_per_residue_class_pdb.py # Script to perform finetuning of PT5 with classification head on PDBBind data
|           ├── pt5_lora_finetuning_per_residue_class_pdb_asd.py # Script to perform finetuning of PT5 previously finetuned on PDBbind data 
|       ├── WeightedCEL/ #
|           ├── pt5_lora_finetuning_per_residue_class_WCEL.py # Script to perform finetuning of PT5 with classification head on ASD data
│
├──  embedding_extraction/
|   ├── Ankh/
|       ├── Ankh_large_ASD_7A.py # Script to train classification head on Ankh embeddings with weighted CE loss 
|   ├── ESM2/
|       ├── esm2_t33_650M_UR50D_ASD_7A.py # Script to train classification head on ESM2_t33_650M embeddings with weighted CE loss
|   ├── ProtT5/
|       ├── ProtT5_ASD_7A.py # Script to train classification head on Prot T5 embeddings with weighted CE loss
│                 
│
├──  data_processing/
|   ├── ASD_data/
|       ├── create_dataset.py # Script to process ASD Dataset and create dataset including protein sequences and allosteric labels
|       ├── prot_seq2Str.py # Script to ensure sequences are in the correct format in dataset
|       ├── drop_duplicate_rows.py # Script to remove any duplicate rows with identical protein sequences and allosteric labels
|       ├── seq_id.py # Script to create train and test datasets by ensuring non of the test sequences have >30% Sequence identity with any of the training set
|   ├── PDB_data/
|       ├── PDBbind_create_dataset.py # Script to extract and format PDBbind data 
|   ├── SSP_data/
|       ├── Make_ssp_dataset.py # Script to generate SSP predictions and create labels for use in multi-task learning

├──  README.md         # Project documentation
```


# Notes on Model Usage

This repository does not contain the actual PLMs (Ankh, ESM2, ProtT5) due to their size and licensing. Instead, the scripts dynamically load them from Hugging Face or other sources. Ensure you have the appropriate access and permissions to use these models.

# Future Work

* Further evaluation on synthetic datasets

