import os
import json
import torch
import numpy as np
from einops import rearrange
from decord import VideoReader
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

VIEWS = {
    'syncam4d': 36,
    'kubric4d': 16,
    'obj4d-10k': 18,
}

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
    video = video[...,0].astype(np.bool)

    # video = rearrange(video, "v f h w c -> v f c h w")
    # 7) if you only need channel-0 (they should all be identical), drop the C axis:
    # video = video[:, :, 0, :, :].astype(np.bool)  # now shape (V, F, H, W)

    # edge_band = compute_edge_band_variable(video)
    
    return video

def process_id_batch(
    video_dir: str,
    out_dir: str,
    ids_batch: list[str],
    num_videos: int
):
    os.makedirs(out_dir, exist_ok=True)
    for seq_id in tqdm(ids_batch):
        arr = load_video(video_dir, seq_id, num_videos)
        for idx in range(arr.shape[0]):
            one_view_video = arr[idx]
            np.save(os.path.join(out_dir, f"{seq_id}_mask_{idx}.npy"), one_view_video)

def main():
    dataset_root = "/fs-computility/llm/shared/konglingdong/data/sets/see4d/sync4d"
    samask_root    = "/fs-computility/llm/shared/konglingdong/data/sets/see4d/sync4d/alldata"
    dataset_list = ['syncam4d', 'kubric4d','obj4d-10k']
    splits       = ['train', 'test']

    # for each (ds, split)
    for ds in dataset_list:
        for split in splits:
            video_dir = os.path.join(dataset_root, ds)
            # load seq IDs and pose‑based neighbor map
            with open(os.path.join(video_dir, f"index_{split}.json")) as f:
                ids = json.load(f)

            num_videos = VIEWS[ds]

            mask_root = f'/fs-computility/llm/shared/konglingdong/data/sets/see4d/sync4d/alldata/{ds}/{split}_mask/mask'

            for video_id in ids:
                for view_id in range(num_videos):
                    mask_file = f'{video_id}_mask_{view_id}.npy'
                    
                    if not os.path.isfile(os.path.join(mask_root, mask_file)):
                    #     print("Found")
                    # else:
                        print(f"File {mask_file} not found in {split} and {ds}")
            
if __name__ == "__main__":
    main()