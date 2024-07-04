# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def rescale_2x(x: torch.Tensor, scale_factor=2):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def resize_to(x: torch.Tensor, tgt_hw: tuple):
    return F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)


def global_avgpool(x: torch.Tensor):
    return x.mean((-1, -2), keepdim=True)


class MaskedGlobalAverage(nn.Module):
    def __init__(self, dim=1):
        super(MaskedGlobalAverage, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"x must be a 4-D tensor, but now, it is {x.ndim}-D.")
        if mask.ndim == 3:  # B, H, W
            mask = mask.unsqueeze(self.dim)
        if mask.ndim != 4:
            raise ValueError(f"mask must be a 4-D or 3-D tensor, but now, it is {mask.ndim}-D.")
        if mask.shape[0] != x.shape[0] or mask.shape[2:] != x.shape[2:]:
            raise ValueError(f"mask must have the shape of [{x.shape[0]}, *, {x.shape[2]}, {x.shape[3]}].")
        x = x.mul(mask).sum(dim=self.dim).div(mask.sum(dim=self.dim))
        return x


class PixelNormalizer(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel.

        Args:
            mean (tuple, optional): the mean value. Defaults to (0.485, 0.456, 0.406).
            std (tuple, optional): the std value. Defaults to (0.229, 0.224, 0.225).
        """
        super().__init__()
        # self.norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        self.register_buffer(name="mean", tensor=torch.Tensor(mean).reshape(3, 1, 1))
        self.register_buffer(name="std", tensor=torch.Tensor(std).reshape(3, 1, 1))

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean.flatten()}, std={self.std.flatten()})"

    def forward(self, x):
        """normalize x by the mean and std values

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor

        Albumentations:

        ```
            mean = np.array(mean, dtype=np.float32)
            mean *= max_pixel_value
            std = np.array(std, dtype=np.float32)
            std *= max_pixel_value
            denominator = np.reciprocal(std, dtype=np.float32)

            img = img.astype(np.float32)
            img -= mean
            img *= denominator
        ```
        """
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x


class LayerNorm2d(nn.Module):
    """
    From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
