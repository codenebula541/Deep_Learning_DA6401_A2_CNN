# Overview
The goal of this assignment is twofold:   
  
(i) Train a CNN model from scratch and learn how to tune the hyperparameters and visualize filters  
(ii) Finetune a pre-trained model   
The link to the wandb project runs:  https://wandb.ai/saurabh541-indian-institute-of-technology-madras/DA6401-A2_1?nw=nwusersaurabh541  and https://wandb.ai/saurabh541-indian-institute-of-technology-madras/DA6401-A2_ET?nw=nwusersaurabh541
The link to the wandb report:  

# Part A: Building and training a CNN model from scratch for classification:  

##  Configurable CNN for Image Classification

A modular and flexible CNN training framework has been implemented for image classification task on inaturalist_12k dataset using PyTorch Lightning and WandB.

## Features

### 🧠  Configurable CNN Model class
- **Adjustable Filter Organization**  
  - Options: `same`, `doubling`, `halving`

- **Customizable Kernel Size & Activations**  
  - Activations: `ReLU`, `GELU`, `SiLU`, `Mish`, `ELU`

- **Batch Normalization (Optional)**  
  - Configurable position: `before` or `after` activation

- **Dropout Support(optional)**  
  - Applied in both convolutional and dense layers

- **Flexible Classifier Head**  
  - Adjustable number of dense units, activation, and dropout

---

### Training Pipeline
- **Dataset Loading**
  - Stratified dataset loading using `torchvision.datasets.ImageFolder` so that from a big train dataset solitting can be done on 80-20 train-val splitting.

- **Data Augmentation & Normalization**
  - Configurable with on/off toggles

- **Callbacks**
  - Early stopping and model checkpointing based on validation metrics

- **Experiment Logging**
  - Integrated with [Weights & Biases (WandB)](https://wandb.ai) for:
    - Full hyperparameter tracking
    - Metric logging
    - Model checkpoint saving

---

### Testing & Visualization
- ** Test Evaluation**
  - Reports overall test set accuracy

- **Per-Class Predictions**
  - Visualizes 3 random predictions per class in a grid where first column shows true class.

---  
###  Structure

| File                         | Purpose                                                                 |
|------------------------------|-------------------------------------------------------------------------|
| `modelClass.py`             | Defines the `FlexibleCNN` model using PyTorch Lightning                 |
| `model_train.py`             | Training pipeline with experiment tracking and WandB integration         |
| `test.py`              | Loads the best model checkpoint and evaluates performance on the test set |
| `sample_image_prediction.py` | Visualizes predictions on random samples and logs them to WandB          |

---  

### Installation  
1. Clone the repository and install dependencies:  


```bash
  git clone https://github.com/yourusername/flexible-cnn.git
  cd flexible-cnn
  pip install torch torchvision pytorch-lightning torchmetrics wandb matplotlib pillow
```
2. Set up your dataset:
   Organize your inaturalist dataset as follows in 80-20 ratio i.e. 80% of train data for training and 20% for validation.  
   Update paths in the scripts at train_dir and val_dir  at model_train.py file.
   
