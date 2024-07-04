import torch
import torch.nn as nn
import torch.nn.functional as F


def seg_loss(logits, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def soft_dice_loss(prob, edge, smooth=1, p=2):
    prob = prob.reshape(prob.shape[0], -1)
    edge = edge.reshape(edge.shape[0], -1)

    num = (prob * edge).sum(dim=1) * 2 + smooth
    den = (prob.pow(p) + edge.pow(p)).sum(dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def edge_dice_loss(logits, edge, smooth=1, p=2):
    if logits.shape[-2:] != edge.shape[-2:]:
        logits = F.interpolate(logits, size=edge.shape[-2:], mode="bilinear", align_corners=False)

    prob = logits.sigmoid()
    return soft_dice_loss(prob, edge, smooth, p)


class GaussianFilter2D(nn.Module):
    def __init__(self, window_size=11, in_channels=1, sigma=1.5) -> None:
        """2D Gaussian Filer

        Args:
            window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
        """
        super().__init__()
        self.window_size = window_size
        if not (window_size % 2 == 1):
            raise ValueError("Window size must be odd.")
        self.in_channels = in_channels
        self.padding = window_size // 2
        self.sigma = sigma
        self.register_buffer(name="gaussian_window2d", tensor=self._get_gaussian_window2d())

    def _get_gaussian_window1d(self):
        sigma2 = self.sigma * self.sigma
        x = torch.arange(-(self.window_size // 2), self.window_size // 2 + 1)
        w = torch.exp(-0.5 * x**2 / sigma2)
        w = w / w.sum()
        return w.reshape(1, 1, self.window_size, 1)

    def _get_gaussian_window2d(self):
        gaussian_window_1d = self._get_gaussian_window1d()
        w = torch.matmul(gaussian_window_1d, gaussian_window_1d.transpose(dim0=-1, dim1=-2))
        w.reshape(1, 1, self.window_size, self.window_size)
        return w.repeat(self.in_channels, 1, 1, 1)

    def forward(self, x):
        x = F.conv2d(input=x, weight=self.gaussian_window2d, padding=self.padding, groups=x.shape[1])
        return x


# from: https://github.com/lartpang/MSSIM.pytorch
class SSIM(nn.Module):
    def __init__(
        self, window_size=11, in_channels=1, sigma=1.5, K1=0.01, K2=0.03, L=1, keep_batch_dim=False, return_log=False
    ):
        """Calculate the mean SSIM (MSSIM) between two 4D tensors.

        Args:
            window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
            K1 (float, optional): K1 of MSSIM. Defaults to 0.01.
            K2 (float, optional): K2 of MSSIM. Defaults to 0.03.
            L (int, optional): The dynamic range of the pixel values (255 for 8-bit grayscale images). Defaults to 1.
            keep_batch_dim (bool, optional): Whether to keep the batch dim. Defaults to False.
            return_log (bool, optional): Whether to return the logarithmic form. Defaults to False.

        ```
            # setting 0: for 4d float tensors with the data range [0, 1] and 1 channel
            ssim_caller = SSIM().cuda()
            # setting 1: for 4d float tensors with the data range [0, 1] and 3 channel
            ssim_caller = SSIM(in_channels=3).cuda()
            # setting 2: for 4d float tensors with the data range [0, 255] and 3 channel
            ssim_caller = SSIM(L=255, in_channels=3).cuda()
            # setting 3: for 4d float tensors with the data range [0, 255] and 3 channel, and return the logarithmic form
            ssim_caller = SSIM(L=255, in_channels=3, return_log=True).cuda()
            # setting 4: for 4d float tensors with the data range [0, 1] and 1 channel,return the logarithmic form, and keep the batch dim
            ssim_caller = SSIM(return_log=True, keep_batch_dim=True).cuda()

            # two 4d tensors
            x = torch.randn(3, 1, 100, 100).cuda()
            y = torch.randn(3, 1, 100, 100).cuda()
            ssim_score_0 = ssim_caller(x, y)
            # or in the fp16 mode (we have fixed the computation progress into the float32 mode to avoid the unexpected result)
            with torch.cuda.amp.autocast(enabled=True):
                ssim_score_1 = ssim_caller(x, y)
            assert torch.isclose(ssim_score_0, ssim_score_1)
        ```
        """
        super().__init__()
        self.window_size = window_size
        self.C1 = (K1 * L) ** 2  # equ 7 in ref1
        self.C2 = (K2 * L) ** 2  # equ 7 in ref1
        self.keep_batch_dim = keep_batch_dim
        self.return_log = return_log

        self.gaussian_filter = GaussianFilter2D(window_size=window_size, in_channels=in_channels, sigma=sigma)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, y):
        """Calculate the mean SSIM (MSSIM) between two 4d tensors.

        Args:
            x (Tensor): 4d tensor
            y (Tensor): 4d tensor

        Returns:
            Tensor: MSSIM
        """
        assert x.shape == y.shape, f"x: {x.shape} and y: {y.shape} must be the same"
        assert x.ndim == y.ndim == 4, f"x: {x.ndim} and y: {y.ndim} must be 4"
        if x.type() != self.gaussian_filter.gaussian_window2d.type():
            x = x.type_as(self.gaussian_filter.gaussian_window2d)
        if y.type() != self.gaussian_filter.gaussian_window2d.type():
            y = y.type_as(self.gaussian_filter.gaussian_window2d)

        mu_x = self.gaussian_filter(x)  # equ 14
        mu_y = self.gaussian_filter(y)  # equ 14
        sigma2_x = self.gaussian_filter(x * x) - mu_x * mu_x  # equ 15
        sigma2_y = self.gaussian_filter(y * y) - mu_y * mu_y  # equ 15
        sigma_xy = self.gaussian_filter(x * y) - mu_x * mu_y  # equ 16

        # equ 13 in ref1
        A1 = 2 * mu_x * mu_y + self.C1
        A2 = 2 * sigma_xy + self.C2
        B1 = mu_x * mu_x + mu_y * mu_y + self.C1
        B2 = sigma2_x + sigma2_y + self.C2
        S = (A1 * A2) / (B1 * B2)

        if self.return_log:
            S = S - S.min()
            S = S / S.max()
            S = -torch.log(S + 1e-8)

        if self.keep_batch_dim:
            return S.mean(dim=(1, 2, 3))
        else:
            return S.mean()


def ssim(
    x, y, *, window_size=11, in_channels=1, sigma=1.5, K1=0.01, K2=0.03, L=1, keep_batch_dim=False, return_log=False
):
    """Calculate the mean SSIM (MSSIM) between two 4D tensors.

    Args:
        x (Tensor): 4d tensor
        y (Tensor): 4d tensor
        window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
        in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
        sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
        K1 (float, optional): K1 of MSSIM. Defaults to 0.01.
        K2 (float, optional): K2 of MSSIM. Defaults to 0.03.
        L (int, optional): The dynamic range of the pixel values (255 for 8-bit grayscale images). Defaults to 1.
        keep_batch_dim (bool, optional): Whether to keep the batch dim. Defaults to False.
        return_log (bool, optional): Whether to return the logarithmic form. Defaults to False.


    Returns:
        Tensor: MSSIM
    """
    ssim_obj = SSIM(
        window_size=window_size,
        in_channels=in_channels,
        sigma=sigma,
        K1=K1,
        K2=K2,
        L=L,
        keep_batch_dim=keep_batch_dim,
        return_log=return_log,
    ).to(device=x.device)
    return ssim_obj(x, y)


def l1_ssim_loss(logits, mask):
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)

    prob = logits.sigmoid()
    l1 = F.l1_loss(input=prob, target=mask, reduction="mean")
    ssiml = 1 - ssim(x=prob, y=mask)
    return l1 + ssiml
