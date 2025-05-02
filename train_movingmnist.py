import time
import math
import yaml
import argparse
from typing import Dict, Any, Type, Callable

import torch
import wandb
import transformers
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.moving_mnist import MovingMNIST
from torchvision.transforms import v2

from models.unet3d import UNet3D
from utils import pprint_yaml, save_checkpoint, load_checkpoint


def train(
    model: Type[nn.Module],
    loader: DataLoader,
    optimizer: Type[torch.optim.Optimizer],
    loss_fn: Callable,
    scaler: torch.GradScaler,
    scheduler: Type[torch.optim.lr_scheduler.LambdaLR],
    device: str,
) -> None:
    """
    Trains a model on a dataloader.
    """
    model.train()
    running_loss, total = 0.0, 0

    for images in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        images = images.permute(0, 2, 1, 3, 4)
        in_seq = images[:, :, :10]
        out_seq = images[:, :, 10:]
        optimizer.zero_grad()

        with torch.autocast(device_type=device):
            outputs = model(in_seq)
            loss = loss_fn(outputs, out_seq)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        total += out_seq.size(0)

    return running_loss / total


@torch.no_grad()
def validate(
    model: Type[nn.Module], loader: DataLoader, loss_fn: Callable, device: str
) -> None:
    """
    Evaluates a model on a dataloader.
    """
    model.eval()
    running_loss, total = 0.0, 0
    for images in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device)
        images = images.permute(0, 2, 1, 3, 4)
        in_seq = images[:, :, :10]
        out_seq = images[:, :, 10:]
        outputs = model(in_seq)
        loss = loss_fn(outputs, out_seq)
        running_loss += loss.item() * images.size(0)
        total += out_seq.size(0)
    return running_loss / total


def main(cfg: Dict[str, Any], wandb_run) -> None:
    """
    Training job orchestrator function. Logs to console output and wandb.
    """
    pprint_yaml(cfg)

    # model-related initializations
    device = cfg["train"]["device"]
    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        features=cfg["model"]["features"],
        up=cfg["model"]["up"],
        residual_blocks=cfg["model"]["residual_blocks"],
        final_activation=nn.Sigmoid(),
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # training-related initializations
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scaler = torch.GradScaler(device)
    ###
    # TODO: Clean this up
    # TODO: Add data aug, regulatization
    # train_loader, val_loader = get_dataloaders(cfg["data"], cfg["train"])
    train_transforms = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomRotation(90, fill=0.0),
            # v2.RandomVerticalFlip(),
            # v2.RandomHorizontalFlip(),
        ]
    )
    val_transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])
    tdata = MovingMNIST("data", download=True, transform=train_transforms)
    vdata = MovingMNIST("data", download=True, transform=val_transforms)
    train_data = Subset(tdata, list(range(0, 9000)))
    val_data = Subset(vdata, list(range(9000, 10000)))
    train_loader = DataLoader(
        train_data, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_data, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=4
    )
    ###
    num_warmup_steps = cfg["train"]["warmup_epochs"] * len(train_loader)
    num_total_steps = cfg["train"]["epochs"] * len(train_loader)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_total_steps
    )

    # training loop
    start_epoch = 0
    best_val_loss = 1e8

    # resume training from checkpoint if specified
    if cfg["train"].get("resume_from_checkpoint"):
        checkpoint_path = cfg["train"]["resume_from_checkpoint"]
        start_epoch = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, cfg["train"]["device"]
        )

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t0 = time.time()
        train_loss = train(
            model, train_loader, optimizer, loss_fn, scaler, scheduler, device
        )
        val_loss = validate(model, val_loader, loss_fn, device)
        epoch_time = time.time() - t0

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        wandb_run.log(
            {
                "train_loss": train_loss,
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
