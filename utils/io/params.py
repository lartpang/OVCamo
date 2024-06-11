# -*- coding: utf-8 -*-

import os

import torch
from torch import nn


def save_weight(save_path: str, model: nn.Module, suffix: str = ""):
    if suffix:
        save_path_wo_suffix, ori_suffix = os.path.splitext(save_path)
        save_path = save_path_wo_suffix + suffix + ori_suffix

    print(f"Saving weight '{save_path}'")
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(model_state, save_path)
    print(f"Saved weight '{save_path}' " f"(only contain the net's weight)")


def load_weight(load_path: str, model: nn.Module, strict=True, prefix="module."):
    assert os.path.exists(load_path), load_path
    params = torch.load(load_path, map_location="cpu")
    params = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in params.items()}
    model.load_state_dict(params, strict=strict)
    print(f"Loaded the weight (only contains the net's weight) from {load_path}")
