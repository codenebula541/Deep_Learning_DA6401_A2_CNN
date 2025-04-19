from torchvision import transforms
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
