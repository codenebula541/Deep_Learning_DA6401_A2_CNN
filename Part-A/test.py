import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from modelClass import FlexibleCNN

# second best performing model configuration having val_acc of 41%
# manual_config = {
#     "base_filters": 32,
#     "filter_organization": "doubling",
#     "kernel_size": 3,
#     "conv_activation": "relu",
#     "batch_norm": True,
#     "bn_position": "after",
#     "conv_dropout": 0,
#     "dense_units": 512,
#     "dense_dropout": 0,
#     "dense_activation": "relu",
#     "learning_rate": 0.0001,
#     "batch_size": 64,
#     "augmentation": False,
#     "weight_decay": 0.0001
# }


# best model configuration having val_acc of 43.6%
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

#  Load model from checkpoint
ckpt_path = "/content/best-epoch=13-val_acc=0.431.ckpt"
model = FlexibleCNN.load_from_checkpoint(ckpt_path, config=manual_config)

# Define transforms (match training)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load test dataset
test_dataset = datasets.ImageFolder(
    root="/content/drive/MyDrive/DL_assignment3_data/inaturalist_12K/test",
    transform=test_transform
)
test_loader = DataLoader(test_dataset, batch_size=manual_config["batch_size"], shuffle=False, num_workers=2)

#  Initialize WandB logger
# wandb_logger = WandbLogger(project="DA6401-A2_ET", log_model=False)
# Run test with WandB logging

trainer = Trainer(accelerator="auto", devices=1 if torch.cuda.is_available() else None)

#trainer = Trainer(logger=wandb_logger, accelerator="auto", devices=1 if torch.cuda.is_available() else None)
trainer.test(model, dataloaders=test_loader)

# Optionally log test accuracy manually (e.g., from metrics dict)
# You can fetch the logged values using: trainer.callback_metrics
test_acc = trainer.callback_metrics.get("test_acc")
if test_acc is not None:
    #wandb_logger.experiment.summary["test_acc"] = test_acc.item()
    print(f"Test Accuracy: {test_acc.item():.4f}")
