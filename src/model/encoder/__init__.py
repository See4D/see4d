from typing import Optional

from .encoder import Encoder
from .encoder_costvolume import EncoderCostVolume, EncoderCostVolumeCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_costvolume import EncoderVisualizerCostVolume
from typing import Tuple, List
from .encoder_depth_anything import EncoderDepthAnythingCfg, EncoderDepthAnything

ENCODERS = {
    "costvolume": (EncoderCostVolume, EncoderVisualizerCostVolume),
    "depth_anything": (EncoderDepthAnything, None),
}

EncoderCfg = EncoderCostVolumeCfg | EncoderDepthAnythingCfg


def get_encoder(cfg: EncoderCfg) -> Tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
