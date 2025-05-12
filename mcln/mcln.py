from typing import List, Union

import torch

_shape_t = Union[int, List[int], torch.Size]


class MCLayerNorm(torch.nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: _shape_t,
        drop_rate: float = 0.2,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        mc_mode: bool = False,
        device=None,
    ):
        # Note: element_wise affine is not passed because of timm
        # zero_ bug when weight is None
        super().__init__(normalized_shape, eps=eps, device=device)

        self.elementwise_affine = elementwise_affine
        self.drop_rate = drop_rate
        self.mc_mode = mc_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # subselect if necessary
        if self.training or self.mc_mode:
            x_sel = subsample(x, self.drop_rate)
        else:
            x_sel = x

        # statistics
        mc_mean = torch.mean(x_sel, dim=-1, keepdim=True)
        mc_var = torch.var(x_sel, dim=-1, keepdim=True, unbiased=False)

        # norm calculation
        x_norm = (x - mc_mean) / torch.sqrt(mc_var + self.eps)

        # Scale and shift
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias

        return x_norm


class MCLayerNorm2d(MCLayerNorm):
    def __init__(
        self,
        normalized_shape: _shape_t,
        drop_rate: float = 0.2,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        mc_mode: bool = False,
        device=None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            drop_rate=drop_rate,
            eps=eps,
            elementwise_affine=elementwise_affine,
            mc_mode=mc_mode,
            device=device,
        )

    def forward(self, x: torch.Tensor):
        # Permute for 2d layer norm, and reverse afterwards
        x = x.permute(0, 2, 3, 1)
        x_norm = super().forward(x)
        x_norm = x_norm.permute(0, 3, 1, 2)

        return x_norm


def subsample(x: torch.Tensor, drop_rate: float) -> torch.Tensor:
    rand = torch.empty_like(x, dtype=torch.float).uniform_()
    idx = rand.topk(int(x.shape[-1] * (1 - drop_rate)), dim=-1).indices

    x_subsampled = torch.gather(x, dim=-1, index=idx)

    return x_subsampled
