import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_tf
from torchvision.utils import make_grid


def plot_results(data_container, save_path, base_size=256, is_rgb=True):
    """Plot the results conresponding to the batched images based on the `make_grid` method from `torchvision`.

    Args:
        data_container (dict): Dict containing data you want to plot.
        save_path (str): Path of the exported image.
    """
    font_cfg = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)

    grids = []
    for subplot_id, (name, data) in enumerate(data_container.items()):
        if data.ndim == 3:
            data = data.unsqueeze(1)

        if subplot_id == 0:
            input_hw = data.shape[-2:]
        else:
            data = F.interpolate(data, size=input_hw, mode="bilinear", align_corners=False)

        grid = make_grid(data, nrow=data.shape[0], padding=2, normalize=False)
        grid = np.array(tv_tf.to_pil_image(grid.float()))
        h, w = grid.shape[:2]
        ratio = base_size / h
        grid = cv2.resize(grid, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)

        (text_w, text_h), baseline = cv2.getTextSize(text=name, **font_cfg)
        text_xy = 20, 20 + text_h // 2 + baseline
        cv2.putText(grid, text=name, org=text_xy, color=(255, 255, 255), **font_cfg)

        grids.append(grid)
    grids = np.concatenate(grids, axis=0)  # H,W,C
    if is_rgb:
        grids = cv2.cvtColor(grids, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, grids)


if __name__ == "__main__":
    data_container = dict(
        image=torch.randn(4, 1, 320, 320),
        mask=torch.randn(4, 1, 320, 320),
        prediction=torch.rand(4, 1, 320, 320),
    )
    plot_results(data_container=data_container, save_path=None)
