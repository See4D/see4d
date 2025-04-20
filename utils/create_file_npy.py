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

def get_two_nearest(sorted_views, cond_view):

    n = len(sorted_views)
    idx = sorted_views.index(cond_view)
    neighbors = []
    # try offsets in order: immediately before, immediately after, two before, two after
    for offset in (-1, +1, -2, +2):
        j = idx + offset
        if 0 <= j < n:
            neighbors.append(sorted_views[j])
        if len(neighbors) == 2:
            break
    return neighbors

def load_video(video_dir, id, num_videos, neighbor_map, 
               samples=10, num_frames_sample=8):
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

    clip_list = []
    for idx in range(samples):
        cond_view = random.randrange(v)
        # nearest neighbors precomputed
        neighbors = neighbor_map[id][cond_view]
        target_view = random.choice(neighbors)
        interval = random.choice([1, 2])

        max_start = f - interval * (num_frames_sample - 1)
        start = random.randrange(max_start)
        # tensorized frame indices
        frames = torch.arange(num_frames_sample, dtype=torch.long) * interval + start

        # advanced indexing: select views then frames
        views = torch.tensor([cond_view, target_view], dtype=torch.long)
        clip = video.index_select(0, views).index_select(1, frames)  # [2, T, c, H, W]
        clip = clip.reshape(-1, c, H, W)  # [2*T, c, H, W]
        clip_list.append(clip)

    video = 255.0*torch.stack(clip_list, dim=0) 

    if num_videos == 36:  # syncam4d
        v, f, c, H, W = video.shape
        video_merged = video.view(v * f, c, H, W)
        # Resize to target resolution (new_H, new_W)
        video_resized = F.interpolate(video_merged, size=(512,512), mode="bilinear", align_corners=False)
        # Reshape back to [v, f, c, new_H, new_W]
        video = video_resized.view(v, f, c, 512, 512)

    return video.numpy().astype(np.uint8)

def process_split(dataset_root, save_root, dataset_name, split, samples=5):
    video_dir = os.path.join(dataset_root, dataset_name)
    num_videos = VIEWS[dataset_name]
    # load IDs & poses
    with open(os.path.join(video_dir, f"index_{split}.json"), 'r') as f:
        ids = json.load(f)
    with open(os.path.join('/data/dylu/project/see4d/data', dataset_name, f"index_{split}_pose.json"), 'r') as f:
        pose_list = json.load(f)
    # build neighbor_map once
    neighbor_map = {
        seq: {
            v: get_two_nearest(pose_list[seq], v)
            for v in range(num_videos)
        }
        for seq in ids
    }

    out_dir = os.path.join(save_root, dataset_name, split)
    os.makedirs(out_dir, exist_ok=True)

    for seq_id in tqdm(ids,
                       desc=f"{dataset_name}/{split}",
                       unit="video",
                       total=len(ids)):
        arr = load_video(video_dir, seq_id, num_videos, neighbor_map, samples)
        np.save(os.path.join(out_dir, f"{seq_id}.npy"), arr)



if __name__ == "__main__":
    dataset_root = "/dataset/yyzhao/Sync4D"
    save_root    = "/dataset/dylu/data/Sync4D"
    # dataset_list = ['syncam4d', 'kubric4d', 'obj4d-10k']
    dataset_list = ['obj4d-10k']
    splits       = ['train']

    # launch one process per (dataset, split)
    jobs = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        for split in splits:
            for ds in dataset_list:
                jobs.append(
                    exe.submit(
                        partial(process_split, dataset_root, save_root, ds, split)
                    )
                )

        # optional: wait for all to finish, you can suppress this bar if you like
        for fut in tqdm(as_completed(jobs),
                        total=len(jobs),
                        desc="All splits"):
            fut.result()