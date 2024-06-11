import torch.nn as nn
from timm.models.layers import LayerNorm2d


def get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "idy":
        return nn.Identity()
    else:
        raise NotImplementedError


class LNConvAct(nn.Sequential):
    def __init__(self, in_planes, out_planes, k, s=1, p=0, d=1, g=1, bias=False, act_name="relu"):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.add_module("ln", LayerNorm2d(in_planes))
        self.add_module("conv", nn.Conv2d(in_planes, out_planes, k, s, p, d, g, bias=bias))
        if act_name is not None:
            self.add_module(name=act_name, module=get_act_fn(act_name=act_name))


class ConvMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        mlp_times: float = 1,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = LayerNorm2d,
        bias: bool = False,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(mlp_times * in_features)

        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=bias)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
