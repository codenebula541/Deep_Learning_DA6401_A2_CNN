import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import wandb
from pytorch_lightning.loggers import WandbLogger

from FlexibleCNN import FlexibleCNN  

# Define training transformations based on config
def get_train_transforms(config):
    transform_list = []

    # Always resize to (224, 224)
    transform_list.append(transforms.Resize((224, 224)))

    # Apply augmentations if enabled
    if config.get("augmentation", False):
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
        ])

    # Convert image to tensor
    transform_list.append(transforms.ToTensor())

    # Apply normalization if enabled
    if config.get("normalize", False):
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))

    # Return composed transform
    return transforms.Compose(transform_list)

# Define validation transformations based on config
def get_val_transforms(config):
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]

    # Apply normalization if enabled
    if config.get("normalize", False):
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))

    return transforms.Compose(transform_list)

# Set train and validation data directories
train_dir = "/content/drive/MyDrive/DL_assignment3_data/inaturalist_12K/stratified_dataset/train"
val_dir = "/content/drive/MyDrive/DL_assignment3_data/inaturalist_12K/stratified_dataset/val"

# Define the manual configuration for the model and training
manual_config = {
    "base_filters": 32,
    "filter_organization": "halving",
    "kernel_size": 3,
    "conv_activation": "mish",
    "batch_norm": True,
    "bn_position": "before",
    "conv_dropout": 0.1,
    "dense_units": 512,
    "dense_dropout": 0.1,
    "dense_activation": "relu",
    "learning_rate": 0.001,
    "batch_size": 32,
    "augmentation": True,
    "weight_decay": 0
}

# Main experiment runner
def run_experiment(use_manual_config=False):
    with wandb.init(config=manual_config if use_manual_config else None) as run:
        config = wandb.config

        # Set run name for manual config
        if use_manual_config:
            wandb.run.name = "ManualRun-" + wandb.util.generate_id()
            wandb.config.update({"custom_info": "Manual config run"}, allow_val_change=True)
        else:
            # Set run name dynamically from config for sweeps
            wandb.run.name = (
                f"F{config.base_filters}-"
                f"{config.filter_organization}-"
                f"{config.conv_activation}-"
                f"BN{config.batch_norm}-{config.bn_position}-"
                f"CD{config.conv_dropout}-"
                f"D{config.dense_units}-DD{config.dense_dropout}-"
                f"Aug{config.augmentation}-"
                f"LR{config.learning_rate:.0e}-"
                f"BS{config.batch_size}-"
                f"WD{config.weight_decay}"
            )

        # Extract model config from wandb config
        model_config = {
            "base_filters": config.base_filters,
            "filter_organization": config.filter_organization,
            "kernel_size": config.kernel_size,
            "conv_activation": config.conv_activation,
            "batch_norm": config.batch_norm,
            "bn_position": config.bn_position,
            "conv_dropout": config.conv_dropout,
            "dense_units": config.dense_units,
            "dense_activation": "relu",
            "dense_dropout": config.dense_dropout,
            "learning_rate": config.learning_rate,
            "augmentation": config.augmentation,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size
        }

        # Set up data loaders
        train_transform = get_train_transforms(model_config)
        val_transform = get_val_transforms(model_config)
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        val_dataset = ImageFolder(val_dir, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        # Define early stopping callback
        early_stop_callback = EarlyStopping(
          monitor="val_acc",
          mode="max",
          patience=3,
          min_delta=0.005,
          verbose=True
        )

        # Define model checkpointing callback
        checkpoint_callback = ModelCheckpoint(
          monitor="val_acc",
          mode="max",
          save_top_k=1,
          filename="best-{epoch:02d}-{val_acc:.3f}",
          save_weights_only=True,
          verbose=True
        )

        # Instantiate model and logger
        model = FlexibleCNN(model_config)
        wandb_logger = WandbLogger(project="DA6401-A2_ET", log_model=True)

        # Create trainer with defined callbacks and logger
        trainer = pl.Trainer(
          max_epochs=20,
          accelerator="auto",
          callbacks=[early_stop_callback, checkpoint_callback],
          logger=wandb_logger,
          enable_checkpointing=True
        )

        # Log hyperparameters to WandB
        wandb_logger.log_hyperparams(model_config)

        # Begin model training
        trainer.fit(model, train_loader, val_loader)

        # Load best model weights
        best_model_path = checkpoint_callback.best_model_path
        model.load_state_dict(torch.load(best_model_path)["state_dict"])
        print("Best model checkpoint saved at:", best_model_path)

        # Log best validation accuracy to WandB summary
        wandb.run.summary["best_val_acc"] = checkpoint_callback.best_model_score

        # Finish WandB run
        run.finish()

# Create a new sweep and start the manual run
sweep_id = wandb.sweep(sweep_config, project="DA6401-A2_ET")
run_experiment(use_manual_config=True)

# Run a single agent iteration for the sweep
wandb.agent("sweep_id", run_experiment, project="DA6401-A2_ET", count=1)

