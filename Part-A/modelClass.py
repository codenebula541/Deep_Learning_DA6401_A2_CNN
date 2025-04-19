import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

class FlexibleCNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Create a list to store convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 3  # Input channels for RGB images

        # Generate a sequence of filter sizes based on the organization strategy
        self.filters = self._generate_filters(
            self.config["base_filters"],
            self.config["filter_organization"]
        )

        # Build each convolutional block
        for i, out_channels in enumerate(self.filters):
            block_layers = []  # List to hold layers of this block

            # Add convolution layer
            block_layers.append(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=config["kernel_size"],
                         padding="same")
            )

            # Add BatchNorm before activation if configured
            if config["batch_norm"] and config["bn_position"] == "before":
                block_layers.append(self._get_normalization(out_channels))

            # Add activation function
            block_layers.append(self._get_activation(config["conv_activation"]))

            # Add BatchNorm after activation if configured
            if config["batch_norm"] and config["bn_position"] == "after":
                block_layers.append(self._get_normalization(out_channels))

            # Add pooling and dropout
            block_layers.extend([
                nn.MaxPool2d(2, 2),
                self._get_dropout(config["conv_dropout"])
            ])

            # Append this block to the list of convolutional blocks
            self.conv_blocks.append(nn.Sequential(*block_layers))
            in_channels = out_channels  # Update input channels for next block

        # Dynamically calculate the size of the flattened feature map
        self.flattened_size = self._calculate_flattened_size()

        # Define the classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, config["dense_units"]),
            self._get_activation(config["dense_activation"]),
            self._get_dropout(config["dense_dropout"]),
            nn.Linear(config["dense_units"], 10)  # Output layer for 10 classes
        )

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Define metrics for training and validation accuracy
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)

    # Generate list of output filters for conv layers
    def _generate_filters(self, base, organization):
        if organization == "same":
            return [base] * 5
        elif organization == "doubling":
            return [base * (2**i) for i in range(5)]
        elif organization == "halving":
            return [max(1, base // (2**i)) for i in range(5)]
        else:
            raise ValueError(f"Unknown filter organization: {organization}")

    # Return the chosen activation function
    def _get_activation(self, name):
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
            "elu": nn.ELU()
        }
        return activations.get(name.lower(), nn.ReLU())

    # Return BatchNorm layer if enabled, else identity
    def _get_normalization(self, channels):
        if self.config["batch_norm"]:
            return nn.BatchNorm2d(channels)
        return nn.Identity()

    # Return dropout layer if rate > 0, else identity
    def _get_dropout(self, rate):
        if rate and rate > 0:
            return nn.Dropout(rate)
        return nn.Identity()

    # Calculate the size of the flattened output after all conv layers
    def _calculate_flattened_size(self):
        dummy_input = torch.randn(1, 3, 224, 224)  # Simulate a dummy input
        with torch.no_grad():
            for block in self.conv_blocks:
                dummy_input = block(dummy_input)
        return dummy_input.flatten(1).size(1)

    # Forward pass through convolutional blocks and classifier
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = x.flatten(1)
        return self.classifier(x)

    # Training step logic
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        acc = self.train_accuracy(preds, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    # Validation step logic
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        acc = self.val_accuracy(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    # Optimizer configuration
    def configure_optimizers(self):
        return torch.optim.Adam(
        self.parameters(),
        lr=self.config["learning_rate"],
        weight_decay=self.config["weight_decay"]
    )

    # Test step logic
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        acc = self.val_accuracy(preds, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)

    # Log training metrics at end of epoch
    def on_train_epoch_end(self):
         train_loss = self.trainer.callback_metrics.get("train_loss")
         train_acc = self.trainer.callback_metrics.get("train_acc")
         print(f"Epoch {self.current_epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Log validation metrics at end of epoch
    def on_validation_epoch_end(self):
         val_loss = self.trainer.callback_metrics.get("val_loss")
         val_acc = self.trainer.callback_metrics.get("val_acc")
         print(f"Epoch {self.current_epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
