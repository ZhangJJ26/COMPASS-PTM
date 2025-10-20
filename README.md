# COMPASS-PTM: A Unified Coarse-to-Fine Multiple PTM Prediction Framework

## Introduction

**COMPASS-PTM** is an interpretable, coarse-to-fine framework that provides an end-to-end solution from proteome-scale, multi-label site profiling to enzyme-substrate pairing. It deciphers the combinatorial "PTM code" to generate testable, mechanistic hypotheses.

![COMPASS_PTM_main](/figures/COMPASS_PTM_main.jpg)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ZhangJJ26/COMPASS-PTM.git
   cd COMPASS-PTM
   ```

2. **Install dependencies:**

   ```bash
   conda env create -f environment.yml
   conda activate compass-ptm
   ```

3. **Download the pretrained model:**

   ```bash
   mkdir checkpoint
   ```

   The pretrained model can be downloaded from the [releases page](https://github.com/ZhangJJ26/COMPASS-PTM/releases) and should be placed in the `checkpoint` directory.

4. **Download the Datasets**

   You can use the mini version in the `data` folder to understand the dataset format. The full dataset will be uploaded later.

## Usage

### Inference

Please refer to the `models/inference.ipynb` file.

### Training

1. **Preprocess**

   This step involves scripts located in the `models/preprocess` directory.

   *   **Preprocess Kinase Embeddings**: Generate embeddings for kinase sequences.
   *   **Preprocess Cross-Talk Prompting Matrix**: Prepare the matrix used for cross-talk prompting.

   ```bash
   cd models/preprocess
   python preprocess_kinase_embeddings.py
   python preprocess_matrix.py
   cd ../.. 
   ```

2. **Training Command**

   ```bash
   cd models
   python train.py
   ```

   **Explanation of `config_pep.py`:**

   This file is used to configure various aspects of the model training and data paths.

   *   **`version`**: Controls the operational stage of the model.
       *   `"stage1"`: For the multi-label PTM site classification task.
       *   `"stage2"`: For the enzyme-substrate pairing task.

   *   **Data (`train_file`, `valid_file`, `test_file`)**: Sets the file paths for training, validation, and test data based on the `version` value.

   *   **Model (`esm2_model`, `mlp_hidden_dims`, `model_checkpoint`, `num_labels`, `v3_feat_dim`)**: Defines the model architecture.
       *   `esm2_model`: The name of the ESM2 model being used (`esm2_t30_150M_UR50D`).
       *   `mlp_hidden_dims`: The dimensions of the hidden layers in the MLP head.
       *   `model_checkpoint`: The identifier for the pretrained ESM2 model on Hugging Face.
       *   `num_labels`: The number of PTM types (for stage1).

   *   **Training (`batch_size`, `learning_rate`, `num_epochs`, `device`, `early_stop`, `patience`, etc.)**: Configures hyperparameters for the training process.
       *   `batch_size`: The batch size for training.
       *   `learning_rate`: The learning rate for the optimizer.
       *   `num_epochs`: The total number of training epochs.
       *   `device`: Specifies whether to use `cuda:0` or `cpu` for computation.
       *   `early_stop` & `patience`: Enables early stopping and sets the number of epochs to wait to prevent overfitting.
       *   `use_wandb` & `wandb_project`: Configuration for experiment tracking with Weights & Biases.

   *   **Checkpoint & Output (`model_save_path`, `checkpoint`)**:
       *   `model_save_path`: The path where the trained model will be saved.
       *   `checkpoint`: The file path for loading pretrained model weights (for stage2).

   *   **Evaluation (`threshold`)**:
       *   `threshold`: The threshold for classifying positive labels in the multi-label classification task.

