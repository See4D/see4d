import os
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from decord import VideoReader

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

def load_video(video_dir, id, target_resolution=(256, 384), num_videos=36, background_color="random"):
    # Construct paths
    video_path = os.path.join(video_dir, 'videos', id + ".mp4")
    # metadata_path = os.path.join(video_dir, 'videos', id + ".json")

    # with open(metadata_path, "r") as f:
    #     metadata = json.load(f)

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
        if background_color == "white":
            bg_color = torch.ones(1,1,1,3, dtype=torch.float32)
        elif background_color == "black":
            bg_color = torch.zeros(1,1,1,3, dtype=torch.float32)
        else:
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
    
    # print(f'Loaded {video.shape[0]} videos with {video.shape[1]} frames each. Resolution: {video.shape[3]}x{video.shape[4]}')

    # Assume video has shape [v, f, c, H, W]
    v, f, c, H, W = video.shape
    # Merge view and frame dimensions
    video_merged = video.view(v * f, c, H, W)
    # Resize to target resolution (new_H, new_W)
    video_resized = F.interpolate(video_merged, size=target_resolution, mode="bilinear", align_corners=False)
    # Reshape back to [v, f, c, new_H, new_W]
    video = video_resized.view(v, f, c, target_resolution[0], target_resolution[1])

    return video

# --- Base Dataset Class for 4D Data ---
class iPhoneDataset(Dataset):
    def __init__(self):
        """
        root: base directory (e.g. "/dataset/yyzhao/Sync4D")
        dataset_name: one of 'syncam4d', 'kubric4d', or 'obj4d-10k'
        split: "train" or "test" (uses index_train.json or index_test.json)
        num_frames_sample: number of contiguous frames to sample per view
        target_resolution: tuple (H, W) to resize frames
        """
        self.root = '/dataset/shared/temp'

        self.imgs = np.load(os.path.join(self.root, 'img.npy'))
        self.warps = np.load(os.path.join(self.root, 'warp.npy'))
        self.masks = np.load(os.path.join(self.root, 'mask.npy'))
        self.target = np.load(os.path.join(self.root, 'target.npy'))
        self.target_resolution = [512, 512]

    def __len__(self):
        return (self.imgs.shape[0])//16

    def __getitem__(self, idx):
        
        idx = idx * 16
        #fetch index every two frames
        index = np.arange(idx,idx+16,2)
        img = self.imgs[index]/255.0
        img = torch.from_numpy(img).float()  # now in [0,1]
        img = img.permute(0, 3, 1, 2)

        warp = self.warps[index]/255.0
        warp = torch.from_numpy(warp).float()  # now in [0,1]
        warp = warp.permute(0, 3, 1, 2)

        mask = self.masks[index]
        mask = torch.from_numpy(mask).float()  # now in [0,1]
        mask = mask[:,None]

        target = self.target[index]/255.0
        target = torch.from_numpy(target).float()
        target = target.permute(0, 3, 1, 2)

        H = img.shape[2]
        W = img.shape[3]

        if H != self.target_resolution[0] or W != self.target_resolution[1]:
            img = F.interpolate(img, size=self.target_resolution, mode="bilinear", align_corners=False)
            warp = F.interpolate(warp, size=self.target_resolution, mode="bilinear", align_corners=False)
            mask = F.interpolate(mask, size=self.target_resolution, mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=self.target_resolution, mode="bilinear", align_corners=False)

        video_sample = torch.concat([img, warp], dim=0)
        # masks = torch.concat([torch.ones_like(mask), mask], dim=0)

        return {
            "video": video_sample,
            "target": target,
            "target_mask": mask,
        }
