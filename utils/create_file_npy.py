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

def load_video(video_dir, id, num_videos):
    # Construct paths
    video_path = os.path.join(video_dir, 'videos', id + ".mp4")

    vr = VideoReader(video_path)
    num_frames = len(vr)
    # num_videos = metadata["num_videos"]
    if num_videos == 36:
        frame_indices = list(range(0, num_frames, 2))
    else:
        frame_indices = list(range(0, num_frames))
    video = vr.get_batch(frame_indices).asnumpy()  # shape: [f, H, W, C], 0-255

    video = torch.from_numpy(video).float() / 255.0  # now in [0,1]

    mask_path = video_path.split(".")[0] + '_mask.mp4'
    if os.path.exists(mask_path):
        mvr = VideoReader(mask_path)
        masks = mvr.get_batch(frame_indices).asnumpy()
        masks = torch.from_numpy(masks).float() / 255.0
        bg_color = torch.randint(0, 256, size=(1,1,1,3), dtype=torch.float32) / 255.0 
        video =  masks * video + (1 - masks) * bg_color

    num_frames, H, W, C = video.shape

    # Process based on dataset
    if num_videos == 36:  # syncam4d
        # For syncam4d, reshape frames to form videos.
        video = video.reshape(num_frames, 6, H//6, 6, W//6, C)
        video = rearrange(video, "f m h n w c -> (m n) f c h w")
    else:
        video_chunks = video.chunk(num_videos, dim=2)  # along width
        video = torch.stack(video_chunks, dim=0)  # [v, f, H, W, C]
        video = rearrange(video, "v f h w c -> v f c h w")
    
    v, f, c, H, W = video.shape
    video = 255.0 * video

    if num_videos == 36:  # syncam4d
        v, f, c, H, W = video.shape
        video_merged = video.view(v * f, c, H, W)
        # Resize to target resolution (new_H, new_W)
        video_resized = F.interpolate(video_merged, size=(512,512), mode="bilinear", align_corners=False)
        # Reshape back to [v, f, c, new_H, new_W]
        video = video_resized.view(v, f, c, 512, 512)

    return video.numpy().astype(np.uint8)

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
            np.save(os.path.join(out_dir, f"{seq_id}_{idx}.npy"), one_view_video)

def main():
    dataset_root = "/fs-computility/llm/shared/konglingdong/data/sync4d"
    save_root    = "/fs-computility/llm/shared/konglingdong/data/sync4d/alldata"
    dataset_list = ['syncam4d', 'kubric4d']
    # dataset_list = ['obj4d-10k']
    splits       = ['train']

    # for each (ds, split)
    for ds in dataset_list:
        for split in splits:
            video_dir = os.path.join(dataset_root, ds)
            out_dir   = os.path.join(save_root, ds, split)

            # load seq IDs and pose‑based neighbor map
            with open(os.path.join(video_dir, f"index_{split}.json")) as f:
                ids = json.load(f)

            num_videos = VIEWS[ds]

            # split IDs into roughly one chunk per CPU
            n_cpus = 128
            id_batches = np.array_split(ids, n_cpus)

            # launch a pool of workers, each handling one batch of IDs
            with ProcessPoolExecutor(max_workers=n_cpus) as exe:
                jobs = []
                for batch in id_batches:
                    jobs.append(
                        exe.submit(
                            partial(
                                process_id_batch,
                                video_dir,
                                out_dir,
                                batch.tolist(),
                                num_videos
                            )
                        )
                    )
                # optional: track progress
                for _ in tqdm(jobs, desc=f"{ds}/{split}", unit="chunk"):
                    _.result()

if __name__ == "__main__":
    main()