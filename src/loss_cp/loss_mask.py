from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch
import torch.nn.functional as F


@dataclass
class LossMaskCfg:
    weight: float


@dataclass
class LossMaskCfgWrapper:
    mask: LossMaskCfg


class LossMask(Loss[LossMaskCfg, LossMaskCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # import ipdb; ipdb.set_trace()
        opacity = prediction.opacity_map # b,v,h,w
        mask = batch["target"]["mask"] # b,v,1,h,w
        has_mask = batch["target"]["has_mask"].bool() # b,
        if has_mask.sum() == 0:
            return torch.tensor(0, dtype=torch.float32, device=opacity.device)
        
        # mask = F.interpolate(mask[:,:,0], opacity.shape[-2:], mode="bilinear", align_corners=False) # b,v,h,w
        delta = opacity[has_mask] - mask[has_mask,:,0]
        return self.cfg.weight * (delta**2).mean()
