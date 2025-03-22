from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch.nn.functional as F

@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # import ipdb; ipdb.set_trace()
        image = batch["target"]["image"]
        mask = batch["target"]["mask"]
        has_mask = batch["target"]["has_mask"].bool() # b,
        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)
        if mask is not None:
            # TODO:
            # it is better to use crop here, but now first use mask
            # mask = F.interpolate(mask[:,:,0], image.shape[-2:], mode="bilinear", align_corners=False)[:,:,None] 
            # Mask should be using the same aug as image, so the shape should be same
            pred = prediction.color * mask 
            gt = image * mask
        else:
            pred = prediction.color
            gt = image
        loss = self.lpips.forward(
            rearrange(pred, "b v c h w -> (b v) c h w"),
            rearrange(gt, "b v c h w -> (b v) c h w"),
            normalize=True,
        )
        return self.cfg.weight * loss.mean()
