import wandb
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# ---- Model config ----
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

ckpt_path = "/content/best-epoch=13-val_acc=0.431.ckpt"
test_root = "/content/drive/MyDrive/DL_assignment3_data/inaturalist_12K/test"

# ---- Load model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlexibleCNN.load_from_checkpoint(ckpt_path, config=manual_config)
model.to(device).eval()

# ---- Dataset and transforms ----
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
test_dataset = datasets.ImageFolder(root=test_root, transform=test_transform)
idx2class = {v: k for k, v in test_dataset.class_to_idx.items()}

# ---- Pick 3 random images per class ----
samples = {}
for idx, cls in idx2class.items():
    paths = [p for p, l in test_dataset.samples if l == idx]
    samples[cls] = random.sample(paths, 3)

# ---- Create 10×5 grid (true label, 3 images, correct count) ----
n_rows = len(samples)
n_cols = 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 30))
fig.tight_layout(pad=3.0)

for i, (cls, paths) in enumerate(samples.items()):
    correct_cnt = 0

    # Column 0: true label
    axes[i, 0].axis('off')
    axes[i, 0].text(0.5, 0.5, cls,
                    ha='center', va='center',
                    fontsize=12, fontweight='bold')

    # Columns 1–3: images and predictions
    for j, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        axes[i, j + 1].imshow(img)
        axes[i, j + 1].axis('off')

        inp = test_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            pred_idx = logits.argmax(dim=1).item()
        pred_name = idx2class[pred_idx]

        if pred_name == cls:
            correct_cnt += 1

        axes[i, j + 1].set_title(f"Pred: {pred_name}", fontsize=10)

    # Column 4: correct/3 summary
    axes[i, 4].axis('off')
    axes[i, 4].text(0.5, 0.5, f"{correct_cnt}/3",
                    ha='center', va='center',
                    fontsize=12, color='green' if correct_cnt == 3 else 'red')

plt.show()

# ---- Log figure to WandB ----
run = wandb.init(
    project="DA6401-A2_ET",
    name="test_predictions_grid",
    reinit=True
)

run.log({
    "predictions_grid": wandb.Image(fig, caption="3 samples per class with true label & correct count")
})

run.finish()
