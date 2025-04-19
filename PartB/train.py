# Function to unfreeze specific blocks
# import the GoogLeNetFineTuner() from the googlrNet_model.py
def unfreeze_layers(pl_module, layer_names):
    """
    Given our LightningModule instance, set requires_grad=True 
    for any submodule whose name is in layer_names.
    """
    for name, module in pl_module.net.named_children():
        if name in layer_names:
            for p in module.parameters():
                p.requires_grad = True

# define a “grid” sweep with exactly one choice per param since here we not required any hyperparameter. only discriminative learning we are doing for tuning purpose


fixed_config = {
    "lr_head":   1e-3,
    "lr_last":   1e-4,
    "lr_middle": 1e-5,
    "lr_early":  1e-6,
}

sweep_config = {      # want to log into the wandb hence using sweep_config mode
    'name': "tune-sweep1",
    "method": "grid",    # grid with one point = no real search
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        k: {"values": [v]} for k, v in fixed_config.items()
    }
}

sweep_id = wandb.sweep(sweep_config, project="DA6401-A2_ET")
print("Created new sweep:", sweep_id)


def train():
    # this auto‑joins the sweep_id above
    wandb.init(project="DA6401-A2_ET")
    cfg = wandb.config

    model = GoogLeNetFineTuner(
        lr_head   = cfg.lr_head,
        lr_last   = cfg.lr_last,
        lr_middle = cfg.lr_middle,
        lr_early  = cfg.lr_early,
    )

    wandb_logger = WandbLogger(project="DA6401-A2_ET")

    # Callbacks (persist across multiple trainer.fit calls)
    early_stop_cb = EarlyStopping(
        monitor="val_acc", mode="max", patience=6, verbose=True
    )
    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
        filename="best-weights-{epoch:02d}-{val_acc:.4f}"
    )
    
    # Phase 1: train head for 5 epochs
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu", devices=1,
        logger=wandb_logger, callbacks=[early_stop_cb, checkpoint_cb],
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    #Phase 2: gradual unfreeze in the same run
    unfreeze_order = ["inception5b", "inception5a","inception4e", "inception4d", "inception4c"]
    for block in unfreeze_order:
        print(f"\n--- Unfreezing block: {block} ---")
        unfreeze_layers(model, [block])
        # re-instantiate Trainer so that configure_optimizers() is called anew
        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="gpu", devices=1,
            logger=wandb_logger, callbacks=[early_stop_cb, checkpoint_cb]
        )
        trainer.fit(model, train_loader, val_loader)
         # Check if early stopped
        if early_stop_cb.stopped_epoch != 0:
            print(f"⏹️ Early stopping triggered at epoch {early_stop_cb.stopped_epoch}")
            break
            
    # Save best model to wandb
    wandb.save(checkpoint_cb.best_model_path)
    print(f"Best model saved at: {checkpoint_cb.best_model_path}")

wandb.agent("sweep-id", function= train, project="DA6401-A2_ET", count=1)
