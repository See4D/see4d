import os
import json
import numpy as np
from einops import rearrange

def load_video(video_dir, id):
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

    # 6) Compute Euclidean distances from the origin
    dists = np.linalg.norm(translations, axis=1)  # shape: (V,)

    # 7) Sort and return both the sorted c2w stack and the original indices
    sorted_idx = np.argsort(dists)       # ascending: closest → farthest

    return sorted_idx.tolist()

dataset_root = "/dataset/yyzhao/Sync4D"
dataset_list = ['syncam4d', 'kubric4d', 'obj4d-10k']

for split in ['train', 'test']:

    save_file = f'index_{split}_pose.json'

    for dataset_name in dataset_list:

        video_dir = os.path.join(dataset_root, dataset_name)

        index_file = f"index_{split}.json"
        index_path = os.path.join(dataset_root, dataset_name, index_file)

        with open(index_path, 'r') as f:
            ids = json.load(f) # list of sequence ids
        pose_index_list = {}
        for id in ids:
            sorted_idx = load_video(video_dir, id)
            pose_index_list[id] = sorted_idx
        
        # Save the pose index
        os.makedirs(os.path.join('data', dataset_name), exist_ok=True)

        save_path = os.path.join('data', dataset_name, save_file)
        with open(save_path, 'w') as f:
            json.dump(pose_index_list, f)
        print(f"Pose index saved to {save_path}")
