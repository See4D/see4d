from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import Gaussians as GaussiansC2W
from .backbone import (
    BackboneMultiview,
)
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding

from .depth_anything.dpt import DepthAnythingGaussians

from typing import Tuple, List


@dataclass
class EncoderDepthAnythingCfg:
    name: Literal["depth_anything"]
    pretrained_model_name_or_path: str | None
    num_surfaces: int
    gaussian_adapter: GaussianAdapterCfg
    downscale_factor: int
    shim_patch_size: int
    gs_feature_dim: int | None = None
    near_disparity: float = 3.0
    apply_bounds_shim: bool = False
    freeze_backbone: bool = False
    ## depth anything configs
    features: int = 256
    use_bn: bool = False
    use_clstoken: bool = False
    localhub: bool = True
    out_channels: List[int] | None = None


class EncoderDepthAnything(Encoder[EncoderDepthAnythingCfg]):
    depth_predictor:  DepthAnythingGaussians
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderDepthAnythingCfg) -> None:
        super().__init__(cfg)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        # NOTE here, +2 means xy offset, +1 means opacity, num_surfaces means the number of gaussians predicted from each pixel
        gaussian_raw_channels = cfg.num_surfaces * (2 + 1 + self.gaussian_adapter.d_in + cfg.gs_feature_dim) if cfg.gs_feature_dim is not None else cfg.num_surfaces * (2 + 1 + self.gaussian_adapter.d_in) 
        
        # multi-view Transformer backbone
        if "hf" in cfg.pretrained_model_name_or_path:
            raise "Please use pytorch hub model version"
        encoder_type = cfg.pretrained_model_name_or_path.split("/")[-1].split("_")[-1][:-2] # remove 14
        self.depth_predictor = DepthAnythingGaussians(
            out_dim=gaussian_raw_channels+cfg.num_surfaces, # gaussian channels with depth, depth should be same with num_surfaces
            encoder=encoder_type,
            gaussians_per_pixel=cfg.num_surfaces,
            features=cfg.features,
            out_channels=cfg.out_channels,
            use_bn=cfg.use_bn,
            use_clstoken=cfg.use_clstoken,
            localhub=cfg.localhub,
        )
        self.depth_predictor.load_from_pretrained(cfg.pretrained_model_name_or_path)
        if cfg.freeze_backbone:
            for param in self.depth_predictor.pretrained.parameters():
                param.requires_grad = False

    def view_to_batch(self, x: Float[Tensor, "batch view ..."]
                      ) -> Float[Tensor, "batchview ..."]:
        return rearrange(x, "batch view ... -> (batch view) () ...")
    
    def batch_to_view(self, x: Float[Tensor, "batchview 1 ..."], num_views: int
                        ) -> Float[Tensor, "batch view ..."]:
        return rearrange(x, "(batch view) () ... -> batch view ...", view=num_views)
    
    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> Tuple[Gaussians, GaussiansC2W]:
        device = context["image"].device
        # import ipdb; ipdb.set_trace()
        b, num_context_views, _, h, w = context["image"].shape
        context["image"] = self.view_to_batch(context["image"])
        context["extrinsics"] = self.view_to_batch(context["extrinsics"])
        context["intrinsics"] = self.view_to_batch(context["intrinsics"])
        
        num_views = context["image"].shape[1]
        depth_gaussian_output = self.depth_predictor(rearrange(context["image"], "b v c h w -> (b v) c h w"))
        # import ipdb; ipdb.set_trace()
        ## NOTE depth should multiply the depth scale
        near = rearrange(context['near'], "b v -> (b v) 1 1 1")
        far = rearrange(context['far'],   "b v -> (b v) 1 1 1")
        depths = depth_gaussian_output["depths"] * (far - near) + near # depths: B,srf,H,W
        depths = rearrange(depths, "(b v) srf h w -> b v (h w) srf 1", v=num_views) # B,V,N,srf,1
        gaussians = rearrange( depth_gaussian_output["raw_gaussians"], "(b v) c srf h w -> b v (h w) srf c ", v=num_views)
        opacities = rearrange( depth_gaussian_output["opacities"], "(b v) c srf h w -> b v (h w) srf c ", v=num_views)

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        # gaussians = rearrange(
        #     raw_gaussians, # b,v,n,c
        #     "... (srf c) -> ... srf c",
        #     srf=self.cfg.num_surfaces,
        # ) # b,v,n,srf,c
        
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        gs_features = None
        if self.cfg.gs_feature_dim is not None:
            gs_features = gaussians[..., -self.cfg.gs_feature_dim:] # b,v,r,srf,c
            gaussians = gaussians[..., :-self.cfg.gs_feature_dim]
            ## TODO check the value of spp
            gs_features = rearrange(gs_features, "(b cv) v r srf c -> b (cv v r srf) c", cv=num_context_views)
            
        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            opacities,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "(b cv) v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "(b cv) v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "(b cv) v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        context["image"] = self.batch_to_view(context["image"], num_context_views)
        context["extrinsics"] = self.batch_to_view(context["extrinsics"], num_context_views)
        context["intrinsics"] = self.batch_to_view(context["intrinsics"], num_context_views)
        
        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        return Gaussians(
            rearrange(
                gaussians.means,
                "(b cv) v r srf spp xyz -> b (cv v r srf spp) xyz", cv=num_context_views,
            ),
            rearrange(
                gaussians.covariances,
                "(b cv) v r srf spp i j -> b (cv v r srf spp) i j", cv=num_context_views,
            ),
            rearrange(
                gaussians.harmonics,
                "(b cv) v r srf spp c d_sh -> b (cv v r srf spp) c d_sh", cv=num_context_views,
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "(b cv) v r srf spp -> b (cv v r srf spp)", cv=num_context_views,
            ),
            features=gs_features,
        ), gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            if self.cfg.apply_bounds_shim:
                _, _, _, h, w = batch["context"]["image"].shape
                near_disparity = self.cfg.near_disparity * min(h, w)
                batch = apply_bounds_shim(batch, near_disparity, 0.5)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
