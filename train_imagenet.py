import time
import math
import yaml
import argparse
from typing import Dict, Any, Type, Callable, Optional

import timm
import torch
import wandb
import transformers
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import DyT, ViT
from data_utils.mixup import get_mixup_fn
from data_utils.imagenet import get_dataloaders
from utils import pprint_yaml, save_checkpoint, load_checkpoint


def train(
    model: Type[nn.Module],
    loader: DataLoader,
    optimizer: Type[torch.optim.Optimizer],
    loss_fn: Callable,
    mixup_fn: Optional[Optional[timm.data.Mixup]],
    scaler: torch.GradScaler,
    scheduler: Type[torch.optim.lr_scheduler.LambdaLR],
    device: str,
) -> None:
    """
    Trains a model on a dataloader.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad()

        with torch.autocast(device_type=device):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        if labels.dim() > 1:  # If using mixup/cutmix (soft labels)
            _, targets = labels.max(1)
            correct += (preds == targets).sum().item()
        else:  # If using regular labels (class indices)
            correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: Type[nn.Module], loader: DataLoader, loss_fn: Callable, device: str
) -> None:
    """
    Evaluates a model on a dataloader.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


# TODO: This shouldn't be in this file
def string_to_norm_layer(norm_layer: str) -> Type[nn.Module]:
    """
    Returns the corresponding torch module based on its name.
    """
    norm_layer = norm_layer.lower()
    if norm_layer == "dyt":
        return DyT
    if norm_layer == "layernorm":
        return nn.LayerNorm
    raise ValueError("Unexpected value for normalization layer.")


def main(cfg: Dict[str, Any], wandb_run) -> None:
    """
    Training job orchestrator function. Logs to console output and wandb.
    """
    pprint_yaml(cfg)

    # model-related initializations
    device = cfg["train"]["device"]
    norm_layer = string_to_norm_layer(cfg["model"].get("norm_layer"))
    model = ViT(
        img_size=cfg["data"]["image_size"],
        patch_size=cfg["model"]["patch_size"],
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        drop=cfg["model"]["dropout"],
        attn_drop=cfg["model"]["attention_dropout"],
        norm_layer=norm_layer,
    ).to(device)

    # training-related initializations
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scaler = torch.GradScaler(device)
    train_loader, val_loader = get_dataloaders(cfg["data"], cfg["train"])
    num_warmup_steps = cfg["train"]["warmup_epochs"] * len(train_loader)
    num_total_steps = cfg["train"]["epochs"] * len(train_loader)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_total_steps
    )
    mixup_fn = get_mixup_fn(cfg["train"].get("mixup"), cfg["model"]["num_classes"])

    # training loop
    start_epoch = 0
    best_val_acc = 0.0

    # resume training from checkpoint if specified
    if cfg["train"].get("resume_from_checkpoint"):
        checkpoint_path = cfg["train"]["resume_from_checkpoint"]
        start_epoch = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, cfg["train"]["device"]
        )

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t0 = time.time()
        train_loss, train_acc = train(
            model, train_loader, optimizer, loss_fn, mixup_fn, scaler, scheduler, device
        )
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        epoch_time = time.time() - t0

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        wandb_run.log(
            {
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch + 1,
            }
        )

        # Save every x epochs
        if (epoch + 1) % cfg["train"]["checkpoint_every_x_epochs"] == 0:
            prefix = (
                f"epoch_{epoch+1:0{math.ceil(math.log10(cfg['train']['epochs']))}}_"
            )
            save_checkpoint(epoch, model, optimizer, scheduler, cfg, prefix)
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            prefix = "best_"
            save_checkpoint(epoch, model, optimizer, scheduler, cfg, prefix)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", help="YAML file specifying run parameters.", required=True
    )
    args = parser.parse_args()

    try:
        with open(args.config_file, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("File not found. Terminating.")
        exit()

    wandb_run = wandb.init(project=cfg["wandb"]["project"], config=cfg)
    main(cfg, wandb_run)
    wandb_run.finish()
