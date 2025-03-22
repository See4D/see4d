from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

import torch.nn.functional as F


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # import ipdb; ipdb.set_trace()
        delta = prediction.color - batch["target"]["image"]
        mask = batch["target"]["mask"]
        if mask is not None:
            # mask = F.interpolate(mask[:,:,0], delta.shape[-2:], mode="bilinear", align_corners=False)[:,:,None] 
            delta = delta * mask
        return self.cfg.weight * (delta**2).mean()
