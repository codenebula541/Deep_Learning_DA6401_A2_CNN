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
---  
### Usage  
#### Training  
Edit model_train.py to set your configuration and data paths, then run:  
```bash
python model_train.py
```
- Supports both manual configuration and WandB sweeps for hyperparameter search.
- Model checkpoints and logs are saved automatically.
#### Testing  
Update the checkpoint path and test directory in model_test.py, then run: like i have saved the checkpoint file name best-epoch=13-val_acc=0.431.ckpt  
```bash
python test.py
```
prints test accuracy and can log results to WandB.  
#### Prediction Visualization
To visualize model predictions on random samples per class and log the grid to WandB:  
```bash
python sample_image_prediction.py
```
### Configuration
bayesian sweeps have been performed for various hyper parameter configurations. The sweep configuration and default configurations of hyperparameters are specficied as follows: e.g.: (below is the my best hyperparameter set (manual_config) found by wandb sweeo run);  
```bash

sweep_config = {
    'name': "cnn-sweep",
    'method': 'bayes',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
    'base_filters': {'values': [16,32, 64]},
    'filter_organization': {'values': ["same", "doubling", "halving"]},
    'conv_activation': {'values': ["relu", "gelu", "mish", "silu"]},
    'batch_norm': {'values': [True, False]},
    'bn_position': {'values': ["before", "after"]},
    'conv_dropout': {'values': [0.0]},
    'dense_units': {'values': [128, 256, 512]},
    'dense_dropout': {'values': [0.0, 0.2, 0.3]},
    'learning_rate': {'values': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]},
    'batch_size': {'values': [32, 64, 128]},
    'augmentation': {'values': [True, False]},
    'weight_decay': {'values': [0, 1e-4, 1e-3]},
    'kernel_size': {'values': [3, 5]}
    }
}

manual_config = {
    "base_filters": 64,
    "filter_organization": "same",
    "kernel_size": 5,
    "conv_activation": "silu",
    "batch_norm": True,
    "bn_position": "after",
    "conv_dropout": 0,
    "dense_units": 256,
    "dense_dropout": 0,
    "dense_activation": "relu",
    "learning_rate": 0.0005,
    "batch_size": 128,
    "augmentation": True,
    "weight_decay": 0.0001
}
```
---
# Part B: Fine tuning a pretrained image classification model.  
For this problem, pretrained models such as GoogLeNet, InceptionV3, ResNet50, VGG, EfficientNetV2, VisionTransformer etc. can used as base models and the user can choose between these models. In present case, i have used GoogLeNet. The user can also choose to freeze all the layers and make them non trainable and only train the newly added dense layers compatible with the number of classes in the dataset. In my case the dense layers were swapped with the output layer having 10 softmax neurons. Gradual Unfreezing (unfreeze layers in stages) + Discriminative Learning Rates (different LRs per layer) till inception4c block was unfreezed for tuning the pre-trained model.



