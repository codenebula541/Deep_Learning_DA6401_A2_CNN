# only use this code when data spliting is required in 80-20 i.e. train-val
import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def create_stratified_dataset(root_dir, val_ratio=0.2, seed=42):
    dataset = ImageFolder(root_dir)
    class_names = dataset.classes
    base_dir = os.path.dirname(root_dir)
    new_root = os.path.join(base_dir, "stratified_dataset")

    os.makedirs(os.path.join(new_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(new_root, "val"), exist_ok=True)

    for class_name in class_names:
        class_path = os.path.join(root_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]
        full_paths = [os.path.join(class_path, f) for f in images]

        train, val = train_test_split(full_paths, test_size=val_ratio, random_state=seed)

        for split, files in [("train", train), ("val", val)]:
            split_path = os.path.join(new_root, split, class_name)
            os.makedirs(split_path, exist_ok=True)
            for f in files:
                shutil.copy(f, os.path.join(split_path, os.path.basename(f)))

    return os.path.join(new_root, "train"), os.path.join(new_root, "val")


