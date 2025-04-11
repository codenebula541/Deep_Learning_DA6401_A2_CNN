import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms

class FlexibleCNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 3  # Input channels (RGB)

        # Generate filter sequence based on organization
        self.filters = self._generate_filters(
            self.config["base_filters"],      #base_filters might be something like 32. first layer filter value
            self.config["filter_organization"] #filter_organization might be "doubling".

        )

        # Create convolutional blocks with configurable BN position
        #
        for i, out_channels in enumerate(self.filters):
            block_layers = []     #dynamically create a list named block_layers, which contains various neural network layers such as convolutional layers, normalization layers, activation functions, pooling layers, and dropout layers.

            # 1. Convolution layer
            block_layers.append(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=config["kernel_size"],
                         padding="same")
            )

            # 2. BatchNorm and Activation order
            if config["batch_norm"] and config["bn_position"] == "before":
                block_layers.append(self._get_normalization(out_channels))

            block_layers.append(self._get_activation(config["conv_activation"]))

            if config["batch_norm"] and config["bn_position"] == "after":
                block_layers.append(self._get_normalization(out_channels))

            # 3. Pooling and Dropout
            block_layers.extend([
                nn.MaxPool2d(2, 2),
                self._get_dropout(config["conv_dropout"])
            ])

            self.conv_blocks.append(nn.Sequential(*block_layers))
            in_channels = out_channels
        #self.conv_blocks is a nn.ModuleList that contains multiple nn.Sequential blocks — and each of those Sequential blocks is made from a custom block_layers list.

        # Calculate flattened size dynamically
        self.flattened_size = self._calculate_flattened_size()    #flattening operation is applied to the output of the last convolutional block (after the dropout layer, if used) in your forward pass before passing it to your dense (classifier) layers

        # Create classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, config["dense_units"]),
            self._get_activation(config["dense_activation"]),
            self._get_dropout(config["dense_dropout"]),
            nn.Linear(config["dense_units"], 10)
        )


        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Accuracy metrics
        
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)



    def _generate_filters(self, base, organization):
        if organization == "same":
            return [base] * 5
        elif organization == "doubling":
            return [base * (2**i) for i in range(5)]
        elif organization == "halving":
            return [max(1, base // (2**i)) for i in range(5)]
        else:
            raise ValueError(f"Unknown filter organization: {organization}")

    def _get_activation(self, name):
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish(),
            "elu": nn.ELU()
        }
        return activations.get(name.lower(), nn.ReLU())

    def _get_normalization(self, channels):
        if self.config["batch_norm"]:
            return nn.BatchNorm2d(channels)
        return nn.Identity()

    def _get_dropout(self, rate):
        if rate and rate > 0:
            return nn.Dropout(rate)
        return nn.Identity()

    def _calculate_flattened_size(self):      #flattening the output of the last convolutional block.
        dummy_input = torch.randn(1, 3, 224, 224)  # Standard image size
        with torch.no_grad():
            for block in self.conv_blocks:
                dummy_input = block(dummy_input)
        return dummy_input.flatten(1).size(1)


    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = x.flatten(1)
        return self.classifier(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        acc = self.train_accuracy(preds, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        acc = self.val_accuracy(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


    # # These hooks are called at the end of each training/validation epoch.
    # def on_train_epoch_end(self):
    #     # Retrieve the logged metrics (averaged over the epoch)
    #     train_loss = self.trainer.callback_metrics.get("train_loss")
    #     train_acc = self.trainer.callback_metrics.get("train_acc")
    #     print(f"Epoch {self.current_epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # def on_validation_epoch_end(self):
    #     val_loss = self.trainer.callback_metrics.get("val_loss")
    #     val_acc = self.trainer.callback_metrics.get("val_acc")
    #     print(f"Epoch {self.current_epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


