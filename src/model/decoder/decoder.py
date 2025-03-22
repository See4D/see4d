from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, List

from jaxtyping import Float
from torch import Tensor, nn

from ...dataset import DatasetCfg
from ..types import Gaussians
from typing import Tuple, List
DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]


@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"]
    viewspace_points: List[Tensor]
    visibility_filter: List[Tensor]
    radii: List[Tensor]
    depth: Float[Tensor, "batch view height width"] | None
    opacity_map: Float[Tensor, "batch view height width"] | None
    features: Float[Tensor, "batch view dim height width"] | None
    


T = TypeVar("T")


class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T
    dataset_cfg: DatasetCfg

    def __init__(self, cfg: T, dataset_cfg: DatasetCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg

    @abstractmethod
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: Tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        pass
