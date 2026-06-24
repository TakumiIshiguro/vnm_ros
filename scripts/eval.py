#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import torch
from torch.utils.data import DataLoader

from train import make_dataset, print_dataset_summary
from vnm_ros.models.model_loader import build_model
from vnm_ros.training.checkpoint import load_training_checkpoint
from vnm_ros.training.trainer import Trainer
from vnm_ros.utils.config import load_runtime_config, package_root, resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    args, _ = parser.parse_known_args()
    cfg = load_runtime_config(args.config_dir)
    train_cfg = cfg["train"]
    model_cfg = dict(cfg["model"])
    if model_cfg["model_type"] == "nomad":
        raise NotImplementedError(
            "NoMaD evaluation is not supported by scripts/eval.py because it "
            "uses a diffusion action head instead of the ViNT supervised head."
        )
    model_cfg.update(
        {
            "context_size": train_cfg["dataset"]["context_size"],
            "len_traj_pred": train_cfg["dataset"]["len_traj_pred"],
            "learn_angle": train_cfg["dataset"]["learn_angle"],
        }
    )
    device_name = train_cfg.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = build_model(model_cfg).to(device)
    load_training_checkpoint(resolve_path(args.checkpoint, package_root()), model, device=device)

    dataset = make_dataset(train_cfg, "test", model_cfg["model_type"])
    print_dataset_summary(dataset, "test")
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["training"]["batch_size"]),
        num_workers=int(train_cfg["training"]["num_workers"]),
        shuffle=False,
    )
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.0),
        scheduler=None,
        device=device,
        run_dir=resolve_path(os.path.join("runs", "eval"), package_root()),
        weights_dir=resolve_path("weights", package_root()),
        config={"model": model_cfg, "train": train_cfg},
        alpha=float(train_cfg["training"]["alpha"]),
        enable_tensorboard=False,
    )
    print(json.dumps(trainer.run_epoch(loader, training=False), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
