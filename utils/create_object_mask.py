import os
import json
import torch
import numpy as np
from einops import rearrange
from decord import VideoReader
import matplotlib.pyplot as plt
import cv2
import random

VIEWS = {
    'syncam4d': 36,
    'kubric4d': 16,
    'obj4d-10k': 18,
}

def visualize_mask_and_edge(mask_vol, edge_vol, view_idx=0, save_path=None, frame_indices=None):
    """
    Display side-by-side the original mask and its edge-band for a given view.

    Args:
        mask_vol      (np.ndarray): shape (V, F, H, W), values 0/1
        edge_vol      (np.ndarray): same shape, 0/1 edge band
        view_idx      (int):        which of the V views to show
        frame_indices (list[int]):  which frames to plot; defaults to up to 8 evenly spaced
    """
    V, F, H, W = mask_vol.shape

    if frame_indices is None:
        # pick up to 8 evenly spaced frames
        frame_indices = np.linspace(0, F-1, min(8, F), dtype=int).tolist()

    n = len(frame_indices)
    fig, axes = plt.subplots(2, n, figsize=(n*2, 4))

    for col, f in enumerate(frame_indices):
        # original mask
        axes[0, col].imshow(mask_vol[view_idx, f], cmap='gray')
        axes[0, col].axis('off')
        # edge band
        axes[1, col].imshow(edge_vol[view_idx, f], cmap='gray')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Mask', rotation=90, labelpad=10)
    axes[1, 0].set_ylabel('Edge Band', rotation=90, labelpad=10)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to: {save_path}")


def compute_edge_band_variable(
    mask_vol: np.ndarray,
    pad_in_range: tuple[int,int]  = (1,50),
    pad_out_range: tuple[int,int] = (1,50),
) -> np.ndarray:
    """
    For each slice in mask_vol (V,F,H,W), pick random pad_in ∈ pad_in_range
    and pad_out ∈ pad_out_range, then build a band of exactly pad_in px
    inside + pad_out px outside the object boundary.

    Returns a uint8 volume (V,F,H,W) with 1s in that band, 0 elsewhere.
    """
    V, F, H, W = mask_vol.shape
    edge_band = np.zeros_like(mask_vol, dtype=np.uint8)

    for v in range(V):
        for f in range(F):
            m = (mask_vol[v, f] > 0).astype(np.uint8)

            pad_in  = random.randint(*pad_in_range)
            pad_out = random.randint(*pad_out_range)

            # structuring elements
            k_in  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*pad_in + 1, 2*pad_in + 1))
            k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*pad_out + 1, 2*pad_out + 1))

            # inner band: foreground minus its eroded version
            inner = cv2.subtract(m, cv2.erode(m, k_in))

            # outer band: dilated version minus the original
            outer = cv2.subtract(cv2.dilate(m, k_out), m)

            # combine
            band = cv2.bitwise_or(inner, outer)
            edge_band[v, f] = band

    return edge_band


def load_video(video_dir, id, num_videos=36):
    # Construct paths
    video_path = os.path.join(video_dir, 'videos', id + ".mp4")

    mask_path = video_path.split(".")[0] + '_mask.mp4'
    mvr = VideoReader(mask_path)
    num_frames = len(mvr)
    frame_indices = list(range(0, num_frames))

    masks = mvr.get_batch(frame_indices).asnumpy()
    masks = masks.astype(np.float32) / 255.0  # shape: (F, H, W*num_videos, C)

    # 4) split the wide frame into `num_videos` views along the width
    #    each chunk has shape (F, H, W, C)
    chunks = np.split(masks, num_videos, axis=2)

    # 5) stack into a single array of shape (V, F, H, W, C)
    video = np.stack(chunks, axis=0)

    video = rearrange(video, "v f h w c -> v f c h w")
    # 7) if you only need channel-0 (they should all be identical), drop the C axis:
    video = video[:, :, 0, :, :].astype(np.bool)  # now shape (V, F, H, W)

    # edge_band = compute_edge_band_variable(video)
    
    return video


dataset_root = "/dataset/yyzhao/Sync4D"
dataset_list = ['obj4d-10k']

save_root = "visualization"

for split in ['train', 'test']:

    for dataset_name in dataset_list:
        
        num_videos = VIEWS[dataset_name]

        video_dir = os.path.join(dataset_root, dataset_name)

        index_file = f"index_{split}.json"
        index_path = os.path.join(dataset_root, dataset_name, index_file)

        with open(index_path, 'r') as f:
            ids = json.load(f) # list of sequence ids

        results = {}
        
        for id in ids:
            save_file = f'pose_matrix_{split}.json'

            # weight_matrix = compute_weight(video_dir, id)
            video = load_video(video_dir, id, num_videos)

            for view_idx in range(video.shape[0]):
                arr = video[view_idx]
                np.save(os.path.join(save_root, f"{id}_mask_{view_idx}.npy"), arr)


            # visualize_mask_and_edge(video, edge, view_idx=0, save_path=save_path)
            # print('====')