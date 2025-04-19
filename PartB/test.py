# import the GoogLeNetFineTuner() from the googlrNet_model.py
# 1. --- WANDB: Join existing sweep run ---
wandb.init(project="DA6401-A2_ET", id="0iqruv2s", resume="allow")  # or "must" to force resume

# 2. --- Load the model from checkpoint i.e. epoch where model performed best ---
ckpt_path = "/kaggle/input/best-weights/best-weights-epoch01-val_acc0.7752.ckpt"
model = GoogLeNetFineTuner.load_from_checkpoint(ckpt_path)
model.eval().cuda()  # Send to GPU

# 3. --- trasform the test iage ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # match ImageNet stats
                         [0.229, 0.224, 0.225])
])

test_data = datasets.ImageFolder(root="/kaggle/input/test-data/test", transform=test_transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

# 4. --- Run inference on test data ---
all_preds, all_targets = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.cuda()
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu()
        all_preds.extend(preds)
        all_targets.extend(y)

# 5. --- Calculate and log test accuracy ---
test_acc = accuracy_score(all_targets, all_preds)
print(f" Test Accuracy: {test_acc:.4f}")

wandb.log({"test_acc": test_acc})
wandb.finish()
