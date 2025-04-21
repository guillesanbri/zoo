import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(cfg):
    """
    Creates train and validation transforms based on the config.
    """
    image_size = cfg.get("image_size", 224)  # Default if not specified
    norm_mean = cfg.get("normalization_mean", (0.5, 0.5, 0.5))  # Example default
    norm_std = cfg.get("normalization_std", (0.5, 0.5, 0.5))  # Example default

    # training transforms
    train_transform_list = [
        transforms.RandomResizedCrop(
            size=(image_size, image_size),
            scale=cfg.get("random_resized_crop_scale", (1.0, 1.0)),
        ),
        transforms.RandomHorizontalFlip(p=cfg.get("random_horizontal_flip_p", 0)),
    ]

    if cfg.get("use_randaugment", False):
        train_transform_list.append(
            transforms.autoaugment.RandAugment(
                num_ops=cfg.get("randaugment_num_ops", 2),
                magnitude=cfg.get("randaugment_magnitude", 9),
            )
        )

    train_transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )

    if cfg.get("use_random_erasing", False):
        train_transform_list.append(
            transforms.RandomErasing(
                p=cfg.get("random_erasing_p", 0.25),
                scale=cfg.get("random_erasing_scale", (0.02, 0.33)),
                ratio=cfg.get("random_erasing_ratio", (0.3, 3.3)),
            )
        )

    train_transform = transforms.Compose(train_transform_list)

    # validation transforms
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )

    print(f"Train Transforms: {train_transform}")
    print(f"Validation Transforms: {val_transform}")

    return train_transform, val_transform


def get_dataloaders(cfg_data, cfg_train):
    """
    Creates and returns the train and validation dataloaders.
    """
    train_dir = cfg_data.get("train_dir")
    val_dir = cfg_data.get("val_dir")

    if not train_dir or not val_dir:
        raise ValueError("train_dir and val_dir must be specified in the data config")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_transform, val_transform = get_transforms(cfg_data)

    # create datasets with ImageFolder
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        print(
            f"Found {len(train_dataset)} training images in {len(train_dataset.classes)} classes."
        )
        print(
            f"Found {len(val_dataset)} validation images in {len(val_dataset.classes)} classes."
        )
        # number of classes should match model output if specified
        if (
            "num_classes" in cfg_data
            and len(train_dataset.classes) != cfg_data["num_classes"]
        ):
            raise ValueError(
                f"Number of classes found ({len(train_dataset.classes)}) differs from config ({cfg_data['num_classes']})"
            )

    except Exception as e:
        print(f"Error creating ImageFolder datasets: {e}")
        raise

    num_workers = cfg_train.get("num_workers", 4)

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg_train["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg_train["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
