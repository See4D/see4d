from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_dinov2 import LossDinoV2, LossDinoV2CfgWrapper
from .loss_mask import LossMask, LossMaskCfgWrapper
from typing import Tuple, List

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossDinoV2CfgWrapper: LossDinoV2,
    LossMaskCfgWrapper: LossMask
    
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossDinoV2CfgWrapper | LossMaskCfgWrapper


def get_losses(cfgs: List[LossCfgWrapper]) -> List[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
