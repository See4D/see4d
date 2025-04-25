import os
import json
import torch
import numpy as np
from einops import rearrange
from decord import VideoReader
import matplotlib.pyplot as plt

VIEWS = {
    'syncam4d': 36,
    'kubric4d': 16,
    'obj4d-10k': 18,
}


def load_video(video_dir, id, num_videos=36, background_color="random"):
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

    return video

def compute_weight(video_dir, id, alpha=0.001, beta=2.0):
    # Construct paths
    metadata_path = os.path.join(video_dir, 'videos', id + ".json")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # 2) Extract poses as an (V, 18) array
    poses = np.array(metadata["cameras"], dtype=np.float32)  # shape: (V, 18)
    V = poses.shape[0]

    # 3) Build world‐to‐camera matrices (w2c) of shape (V, 4, 4)
    w2c = np.tile(np.eye(4, dtype=np.float32), (V, 1, 1))
    # The last 12 elements of each pose are the flattened 3×4 [R|t] matrix
    w2c[:, :3, :] = poses[:, 6:].reshape(V, 3, 4)

    # 5) Extract translation vectors (the last column of each 4×4)
    translations = w2c[:, :3, 3]          # shape: (V, 3)
    rotations    = w2c[:, :3, :3]     # (V, 3, 3)

    # 3) Compute W matrix
    W = np.zeros((V, V), dtype=np.float32)
    for i in range(V):
        t_i = translations[i]
        R_i = rotations[i]
        for j in range(V):
            # Translation distance
            dt = np.linalg.norm(t_i - translations[j])  # :contentReference[oaicite:5]{index=5}
            # Rotation distance
            R_diff = R_i @ rotations[j].T               # :contentReference[oaicite:6]{index=6}
            cos_arg = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            dr = np.arccos(cos_arg)                      # :contentReference[oaicite:7]{index=7}
            # Weighted sum
            W[i, j] = alpha * dt + beta * dr            # :contentReference[oaicite:8]{index=8}
    return W

dataset_root = "/dataset/yyzhao/Sync4D"
dataset_list = ['syncam4d', 'kubric4d', 'obj4d-10k']

for split in ['train', 'test']:

    for dataset_name in dataset_list:

        video_dir = os.path.join(dataset_root, dataset_name)

        index_file = f"index_{split}.json"
        index_path = os.path.join(dataset_root, dataset_name, index_file)

        with open(index_path, 'r') as f:
            ids = json.load(f) # list of sequence ids

        results = {}
        
        for id in ids:
            save_file = f'pose_matrix_{split}.json'

            weight_matrix = compute_weight(video_dir, id)
            # video = load_video(video_dir, id, num_videos)
            nn_idx = np.argsort(weight_matrix, axis=1)

            results[id] = nn_idx


        # 3) serialize to one big JSON file
        out_name = f"pose_matrix_{split}_{dataset_name}.npz"
        out_path = os.path.join('data', out_name)
        np.savez_compressed(out_path, **results)
        print(f"Saved arrays to {out_path}")
            
            # V = weight_matrix.shape[0]

            # fig, axes = plt.subplots(2, V, figsize=(V * 2, 4))  # skip self (0th)

            # for i in range(V):

            #     img_orig = video[i,0]
            #     img_nn = video[nn_idx[i],0]

            #     # move channel to last axis for imshow
            #     img_orig = np.transpose(img_orig, (1, 2, 0))  # (H, W, C)
            #     img_nn   = np.transpose(img_nn,   (1, 2, 0))

            #     ax0 = axes[0, i]
            #     ax1 = axes[1, i]
            #     ax0.imshow(img_orig); ax0.axis('off')
            #     ax1.imshow(img_nn);   ax1.axis('off')

            # axes[0, 0].set_ylabel("Original", fontsize=12)
            # axes[1, 0].set_ylabel("Nearest Neighbor", fontsize=12)

            # plt.tight_layout()
            # # ——— save instead of show ———
            # out_path = os.path.join(output_dir, f"viz_{id}.png")
            # fig.savefig(out_path, dpi=200, bbox_inches="tight")
            # plt.close(fig)  # free memory