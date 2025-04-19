import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torchvision.models import googlenet, GoogLeNet_Weights

from sklearn.metrics import accuracy_score


# Cell 3: LightningModule Definition
class GoogLeNetFineTuner(pl.LightningModule):
    def __init__(self, lr_head=1e-3, lr_last=1e-4, lr_middle=1e-5, lr_early=1e-6): 
    #              lr_head=1e-3, lr_last=1e-4, 
    #              lr_middle=1e-5, lr_early=1e-6):
        super().__init__()
        self.save_hyperparameters()

    # Pull learning rates from wandb.config (set in train_fixed())
        #config = wandb.config
        self.lr_head   = lr_head
        self.lr_last   = lr_last
        self.lr_middle = lr_middle
        self.lr_early  = lr_early


    

        # 1) load pre-trained GoogLeNet + replace head
        self.net = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        self.net.fc = nn.Linear(self.net.fc.in_features, 10)

        # 2) freeze all, then unfreeze head
        for p in self.net.parameters(): p.requires_grad = False
        for p in self.net.fc.parameters():  p.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x): 
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc  = accuracy_score(y.cpu(), logits.argmax(dim=1).cpu())
        self.log("train_loss", loss)
        self.log("train_acc",  acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc  = accuracy_score(y.cpu(), logits.argmax(dim=1).cpu())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",  acc,  prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}


    def configure_optimizers(self):
        # discriminative LR parameter groups
        groups = [
                {"params": self.net.inception4c.parameters(), "lr": self.hparams.lr_early},
                {"params": self.net.inception4d.parameters(), "lr": self.hparams.lr_early},
                {"params": self.net.inception4e.parameters(), "lr": self.hparams.lr_middle},
                {"params": self.net.inception5a.parameters(), "lr": self.hparams.lr_last},
                {"params": self.net.inception5b.parameters(), "lr": self.hparams.lr_last},
                {"params": self.net.fc.parameters(),          "lr": self.hparams.lr_head},
        ]

        opt = torch.optim.Adam(groups, weight_decay=1e-4)
        return opt
