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

#import torchmetrics

def get_train_transforms(config):
    transform_list = []

    # Always resize to (224, 224)
    transform_list.append(transforms.Resize((224, 224)))

    # If augmentation is enabled in the config, add augmentation transforms
    if config.get("augmentation", False):
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
        ])

    # Convert image to Tensor (normalizes pixel values to [0, 1])
    transform_list.append(transforms.ToTensor())

    # If normalization is enabled, add normalization transform (commonly used for pretrained networks)
    if config.get("normalize", False):
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))

    # Combine all transformations into one pipeline
    return transforms.Compose(transform_list)


def get_val_transforms(config):
    transform_list = [
        # Resize all validation images to (224, 224)
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]

    # Add normalization if enabled
    if config.get("normalize", False):
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))

    return transforms.Compose(transform_list)


# Set the train and validation directories manually:
train_dir = "/content/drive/MyDrive/DL_assignment3_data/inaturalist_12K/stratified_dataset/train"
val_dir = "/content/drive/MyDrive/DL_assignment3_data/inaturalist_12K/stratified_dataset/val"

# Set your manual config here
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



def run_experiment(use_manual_config=False):
    with wandb.init(config=manual_config if use_manual_config else None) as run:
        config = wandb.config

        if use_manual_config:
            wandb.run.name = "ManualRun-" + wandb.util.generate_id()
            wandb.config.update({"custom_info": "Manual config run"}, allow_val_change=True)
        else:
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

        # Data and model setup...
        train_transform = get_train_transforms(model_config)
        val_transform = get_val_transforms(model_config)
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        val_dataset = ImageFolder(val_dir, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        # Set up EarlyStopping and ModelCheckpoint callbacks

        early_stop_callback = EarlyStopping(
          monitor="val_acc",    # This should match the metric name logged in your validation_step
          mode="max",           # We want to maximize validation accuracy
          patience=3,           # Number of epochs to wait for improvement
          min_delta=0.005,      # Minimum improvement threshold
          verbose=True
        )

        checkpoint_callback = ModelCheckpoint(
          monitor="val_acc",        # Monitor the same metric as EarlyStopping
          mode="max",
          save_top_k=1,             # Only keep the best model checkpoint
          filename="best-{epoch:02d}-{val_acc:.3f}",
          save_weights_only=True,   # Save only the weights (optional)
          verbose=True,
          #restore_on_train_end=True # If supported, automatically load best weights at the end
        )

        model = FlexibleCNN(model_config)

        wandb_logger = WandbLogger(project="DA6401-A2_ET", log_model=True)

        trainer = pl.Trainer(
          max_epochs=20,
          accelerator="auto",
          callbacks=[early_stop_callback, checkpoint_callback],
          logger=wandb_logger,
          enable_checkpointing=True
        )

        wandb_logger.log_hyperparams(model_config)
        trainer.fit(model, train_loader, val_loader)




        # Load best weights manually
        best_model_path = checkpoint_callback.best_model_path
        model.load_state_dict(torch.load(best_model_path)["state_dict"])
        print("Best model checkpoint saved at:", best_model_path)

        # Optionally, update the wandb run summary with the best validation accuracy.
        wandb.run.summary["best_val_acc"] = checkpoint_callback.best_model_score

        run.finish()
        
sweep_id = wandb.sweep(sweep_config, project="DA6401-A2_ET")
run_experiment(use_manual_config=True)
wandb.agent("sweep_id", run_experiment, project="DA6401-A2_ET", count=1)
