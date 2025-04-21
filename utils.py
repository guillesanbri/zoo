import os
import yaml
from typing import Type, Any, Dict

import torch
import torch.nn as nn


def pprint_yaml(y: Dict[str, Any]) -> None:
    """
    Pretty prints a dictionary loaded from a YAML file.
    """
    print(yaml.dump(y, sort_keys=False, default_flow_style=False, indent=2))


def save_checkpoint(
    epoch: int,
    model: Type[nn.Module],
    optimizer: Type[torch.optim.Optimizer],
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    cfg: Dict[str, Any],
    prefix: str,
):
    """
    Stores epoch, model state, optimizer state, scheduler state
    and config in a pth checkpoint. Output name is {prefix}checkpoint.pth
    """
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg,
    }

    output_dir = cfg["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, prefix + "checkpoint.pth")
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint to {filename}")


def load_checkpoint(
    checkpoint_path: str,
    model: Type[nn.Module],
    optimizer: Type[torch.optim.Optimizer],
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: str,
) -> int:
    """
    Loads epoch, model state, optimizer state and scheduler state
    from a pth checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        print(
            f"Checkpoint path {checkpoint_path} does not exist. Starting from scratch."
        )
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded model state dict.")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Loaded optimizer state dict.")
    else:
        print("Optimizer state not loaded (not provided or not in checkpoint).")

    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"]
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("Loaded scheduler state dict.")
    else:
        print(
            "Scheduler state not loaded (not provided, not in checkpoint, or was None)."
        )

    start_epoch = checkpoint.get("epoch", 0)
    # loaded_cfg = checkpoint.get('config')

    print(
        f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {start_epoch}."
    )
    return start_epoch
