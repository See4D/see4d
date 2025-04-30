import numpy as np
import torch
from einops import reduce
from jaxtyping import Float
import lpips
# from skimage.metrics import structural_similarity
from torch import Tensor
from skimage.metrics import structural_similarity as ssim_fn

def compute_psnr_np(
    ground_truth: np.ndarray,  # shape: (B, C, H, W), values in [0,1]
    predicted:    np.ndarray,  # shape: (B, C, H, W), values in [0,1]
    eps: float = 1e-8
) -> np.ndarray:               # returns shape: (B,)
    """
    Compute per-image PSNR in dB between ground_truth and predicted.

    PSNR = 10 * log10(1 / (MSE + eps)), assuming images are in [0,1].
    """
    # 1) Clamp to [0,1]
    gt   = np.clip(ground_truth, 0.0, 1.0)
    pred = np.clip(predicted,  0.0, 1.0)

    # 2) Compute mean squared error over channels and spatial dims
    #    axis=(1,2,3) collapses C, H, W, leaving shape (B,)
    mse = np.mean((gt - pred) ** 2, axis=(1, 2, 3))

    # 3) PSNR in decibels
    psnr = 10.0 * np.log10(1.0 / (mse + eps))
    return psnr


def compute_ssim_np(
    ground_truth: np.ndarray,  # (B, C, H, W), values in [0,1]
    predicted:    np.ndarray,  # (B, C, H, W), values in [0,1]
    win_size:     int   = 11,
    gaussian_weights: bool = True,
    data_range:   float = 1.0
) -> np.ndarray:               # returns (B,)
    """
    Compute SSIM for each image in a batch.
    SSIM is computed channel-wise (multichannel) and averaged internally by skimage.

    Args:
        ground_truth: float array in [0,1], shape (B, C, H, W)
        predicted:    float array in [0,1], shape (B, C, H, W)
        win_size:     size of the sliding window (odd integer)
        gaussian_weights: whether to use a Gaussian window
        data_range:   the dynamic range of the data (1.0 for [0,1])

    Returns:
        A 1D array of SSIM scores (in [-1,1] or [0,1] depending on data)
    """
    assert ground_truth.shape == predicted.shape, "Input shapes must match"
    B, C, H, W = ground_truth.shape

    ssim_vals = np.empty(B, dtype=np.float32)
    for i in range(B):
        # each call expects arrays of shape (C, H, W) and channel_axis=0
        gt_i  = ground_truth[i]
        pred_i = predicted[i]

        ssim_vals[i] = ssim_fn(
            gt_i, pred_i,
            win_size=win_size,
            gaussian_weights=gaussian_weights,
            channel_axis=0,
            data_range=data_range
        )
    return ssim_vals

# 1) Instantiate once, on CPU
# _lpips_model = lpips.LPIPS(net='alex').eval()

def compute_lpips_np(
    ground_truth:   np.ndarray,  # (B, C, H, W) or (C, H, W), values in [0,1]
    predicted: np.ndarray,   # same shape as gt
    device
) -> np.ndarray:       # returns (B,)
    """
    Compute LPIPS distances on CPU without tracking gradients.
    """
    # 1) Instantiate once, on CPU
    _lpips_model = lpips.LPIPS(net='vgg').to(device).eval()

    # 2) Ensure a batch dimension
    if ground_truth.ndim == 3:
        ground_truth   = ground_truth[None, ...]
        predicted = predicted[None, ...]

    # 3) To torch tensors and normalize to [-1, +1]
    gt_t   = torch.from_numpy(ground_truth).float()   * 2.0 - 1.0
    pred_t = torch.from_numpy(predicted).float() * 2.0 - 1.0

    # 4) Move to the same device as the model
    gt_t   = gt_t.to(device)
    pred_t = pred_t.to(device)

    # 4) Run the model under no_grad (all on CPU)
    with torch.no_grad():
        dist = _lpips_model(gt_t, pred_t)  # shape (B,1,1,1)

    # 5) Squeeze and convert back to NumPy
    return dist.view(dist.size(0)).cpu().numpy().astype(np.float32)

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


# @cache
# def get_lpips(device: torch.device) -> LPIPS:
#     return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        ssim_fn(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)