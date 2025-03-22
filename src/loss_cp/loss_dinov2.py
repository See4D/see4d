from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from transformers import AutoImageProcessor, AutoModel
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

import torch.nn.functional as F

from typing import List, Tuple


@dataclass
class LossDinoV2Cfg:
    weight: float
    apply_after_step: int
    model_verison: str = 'facebook/dinov2-base'


@dataclass
class LossDinoV2CfgWrapper:
    dino: LossDinoV2Cfg

def get_parameter_device(parameter: torch.nn.Module):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device
    
class LossDinoV2(Loss[LossDinoV2Cfg, LossDinoV2CfgWrapper]):
    model: AutoModel

    def __init__(self, cfg: LossDinoV2CfgWrapper) -> None:
        super().__init__(cfg)

        self.processor = AutoImageProcessor.from_pretrained(self.cfg.model_verison)
        self.model = AutoModel.from_pretrained(self.cfg.model_verison)
        convert_to_buffer(self.model, persistent=False)

        
    # @property
    # def device(self, ):
    #     return get_parameter_device(self.model)
    
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"]

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step or self.cfg.weight == 0:
            return torch.tensor(0, dtype=torch.float32, device=image.device)
        # import ipdb; ipdb.set_trace() ## TODO make sure the input is 0-1
        inputs = self.processor(images=rearrange(image, "b v c h w -> (b v) c h w"), return_tensors="pt", do_rescale=False)
        with torch.no_grad():
            predict_dino = self.model(**inputs.to(prediction.features.device, prediction.features.dtype)).last_hidden_state[:,1:] # B,256,C
            
        patch_size = int(predict_dino.size(1) ** 0.5) # 16
        predict_dino = rearrange(predict_dino, "b (h w) c -> b c h w", h=patch_size, w=patch_size) # B,C,H,W
        prediction = F.interpolate(rearrange(prediction.features, "b v c h w -> (b v) c h w"), size=(patch_size, patch_size), mode="bilinear")
        
        loss = F.mse_loss(prediction, predict_dino)

        return self.cfg.weight * loss.mean()
