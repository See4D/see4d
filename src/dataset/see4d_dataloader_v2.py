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
class Base4DDataset(Dataset):
    def __init__(self, root, dataset_name, split="train", num_frames_sample=8, target_resolution=(256,384)):
        """
        root: base directory (e.g. "/dataset/yyzhao/Sync4D")
        dataset_name: one of 'syncam4d', 'kubric4d', or 'obj4d-10k'
        split: "train" or "test" (uses index_train.json or index_test.json)
        num_frames_sample: number of contiguous frames to sample per view
        target_resolution: tuple (H, W) to resize frames
        """
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        index_file = f"index_{split}.json"
        self.index_file = os.path.join(root, dataset_name, index_file)
        with open(self.index_file, 'r') as f:
            self.ids = json.load(f) # list of sequence ids
        # print('======================================================load index file====')
        # self.pose_list_file = os.path.join('data', dataset_name, f'index_{split}_pose.json')
        # with open(self.pose_list_file, 'r') as f:
        #     self.pose_list = json.load(f) # list of sequence ids
        self.num_frames_sample = num_frames_sample
        self.target_resolution = target_resolution
        self.num_videos = VIEWS[dataset_name]

        # self.neighbor_map = {}
        # for seq in self.ids:
        #     sorted_views = self.pose_list[seq]
        #     self.neighbor_map[seq] = {}
        #     for v in range(self.num_videos):
        #         self.neighbor_map[seq][v] = get_two_nearest(sorted_views, v)

        self.npy_dataset_root = os.path.join(root, 'npy')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        seq = self.ids[idx]
        # load & cache

        video_dir = os.path.join(self.npy_dataset_root, self.dataset_name, self.split, seq+'.npy')
        # print(f'we load data from {video_dir}')

        video = np.load(video_dir)/255.0
        video = torch.from_numpy(video).float()  # now in [0,1]

        v, f, c, H, W = video.shape
        cond_view = random.randrange(v)

        video_sample = video[cond_view]  # [f, c, H, W]

        if H != self.target_resolution[0] or W != self.target_resolution[1]:
            video_sample = F.interpolate(video_sample, size=self.target_resolution, mode="bilinear", align_corners=False)


        # # nearest neighbors precomputed
        # neighbors = self.neighbor_map[seq][cond_view]
        # target_view = random.choice(neighbors)
        # interval = random.choice([1, 2])

        # max_start = f - interval * (self.num_frames_sample - 1)
        # start = random.randrange(max_start)
        # # tensorized frame indices
        # frames = torch.arange(self.num_frames_sample, dtype=torch.long) * interval + start

        # # advanced indexing: select views then frames
        # views = torch.tensor([cond_view, target_view], dtype=torch.long)
        # clip = video.index_select(0, views).index_select(1, frames)  # [2, T, c, H, W]
        # clip = clip.reshape(-1, c, H, W)  # [2*T, c, H, W]

        return {
            "video": video_sample
        }


    # def __getitem__(self, idx):
    #     sequence_id = self.ids[idx]
    #     video_dir = os.path.join(self.root, self.dataset_name)
    #     video = load_video(video_dir, sequence_id, target_resolution=self.target_resolution, num_videos=self.num_videos)
    #     # video: [v, f, c, H, W]
    #     v, f, c, H, W = video.shape

    #     cond_view = random.randrange(v)

    #     sorted_views = self.pose_list[sequence_id]

    #     nearest4 = get_two_nearest(sorted_views, cond_view)

    #     target_view = random.choice(nearest4)
    #     interval = random.choice([1, 2])
        
    #     max_start = f - interval * (self.num_frames_sample - 1)
    #     start = random.randrange(max_start)
    #     frames = [start + k * interval for k in range(self.num_frames_sample)]

    #     views = [cond_view, target_view]
    #     clip = video[views][:, frames]       # [2, T, C, H, W]
    #     clip = clip.reshape(-1, c, H, W)        # [2*T, C, H, W]

    #     sample = {
    #         "video": clip,
    #         "view_indices": views,
    #         "frame_indices": frames
    #     }
    #     return sample

# --- Subclasses for Each Dataset ---
class Syncam4DDataset(Base4DDataset):
    def __init__(self, root, split="train", num_frames_sample=8, target_resolution=(512,512)):
        # For syncam4d, assume 36 views and group_size=9.
        super().__init__(root, 'syncam4d', split, num_frames_sample, target_resolution=target_resolution)

class Kubric4DDataset(Base4DDataset):
    def __init__(self, root, split="train", num_frames_sample=8, target_resolution=(256,384)):
        # For kubric4d, assume 16 views and choose group_size accordingly (e.g., 6).
        super().__init__(root, 'kubric4d', split, num_frames_sample, target_resolution=target_resolution)

class Obj4D10kDataset(Base4DDataset):
    def __init__(self, root, split="train", num_frames_sample=8, target_resolution=(256,384)):
        # For obj4d-10k, assume 18 views and use group_size=6.
        super().__init__(root, 'obj4d-10k', split, num_frames_sample, target_resolution=target_resolution)

# --- Combined Dataset ---
def get_combined_dataset(root, split="train", num_frames_sample=8, target_resolution=(512,512)):
    dataset_syncam = Syncam4DDataset(root, split, num_frames_sample, target_resolution)
    dataset_kubric = Kubric4DDataset(root, split, num_frames_sample, target_resolution)
    dataset_obj4d = Obj4D10kDataset(root, split, num_frames_sample, target_resolution)
    # combined = ConcatDataset([dataset_syncam, dataset_kubric, dataset_obj4d])
    return [dataset_syncam, dataset_kubric, dataset_obj4d]

# # --- Visualization Function ---
# def visualize_sample(sample, save_path="visualization.png", view_indices=None, frame_indices=None):
#     # sample["video"]: [batch, num_views, num_frames, c, H, W]
#     # Here we use the first batch item.
#     # video = sample[0]
#     num_views, num_frames, c, H, W = sample.shape
#     fig, axes = plt.subplots(num_views, num_frames, figsize=(num_frames * 2, num_views * 2))
#     if num_views == 1:
#         axes = [axes]
#     for i in range(num_views):
#         for j in range(num_frames):
#             frame = sample[i, j].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
#             axes[i][j].imshow(frame)
#             axes[i][j].axis("off")
#     plt.suptitle(f"Views: {view_indices}\nFrames: {frame_indices}")
#     # Save the figure to the given path
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.close(fig)  # close the figure to free memory


def visualize_sample(
    sample,
    save_path="visualization.png",
    view_indices=None,
    frame_indices=None,
    warp_images=None,
    images_predict_batch=None,
):
    """
    Visualizes a batch of video samples with optional extra rows for warp images and predicted images.
    
    Args:
        sample (torch.Tensor): Tensor with shape [num_views, num_frames, c, H, W].
        save_path (str): Path to save the visualization.
        view_indices: Optional indices to include in the title.
        frame_indices: Optional indices to include in the title.
        warp_images (list or None): If provided, a list of images (e.g. PIL or numpy arrays) for the warp row.
        images_predict_batch (list or None): If provided, a list of predicted images (e.g. PIL or numpy arrays)
            for the prediction row.
    """
    num_frames, c, H, W = sample.shape

    # Calculate extra rows: one for warp, one for predictions, if provided.
    extra_rows = (1 if warp_images is not None else 0) + (1 if images_predict_batch is not None else 0)
    total_rows = 1 + extra_rows

    # Create subplots grid.
    fig, axes = plt.subplots(total_rows, num_frames, figsize=(num_frames * 2, total_rows * 2))
    
    # Ensure axes is a 2D array even if there's only one row.
    if total_rows == 1:
        axes = np.array([axes])
    
    current_row = 0
    # Plot sample frames (first num_views rows).
    for j in range(num_frames):
        frame = sample[j].permute(1, 2, 0).numpy()  # shape: H x W x c
        axes[current_row][j].imshow(frame)
        axes[current_row][j].axis("off")
    
    current_row += 1  # start extra rows after the sample rows

    # Plot warp images if provided.
    if warp_images is not None:
        # Expecting warp_images to be a list of images with length equal to num_frames.
        for j in range(num_frames):
            img = warp_images[j]
            img = img.permute(1, 2, 0).numpy()
            axes[current_row][j].imshow(img)
            axes[current_row][j].axis("off")
        current_row += 1

    # Plot predicted images if provided.
    if images_predict_batch is not None:
        # Expecting images_predict_batch to be a list of images with length equal to num_frames.
        for j in range(num_frames):
            img = images_predict_batch[j]
            if hasattr(img, "convert"):
                img = np.array(img)/255.0
                # img = np.array(img)
            axes[current_row][j].imshow(img)
            axes[current_row][j].axis("off")
        current_row += 1

    # # Plot predicted images if provided.
    # if images_predict_batch is not None:
    #     # Expecting images_predict_batch to be a list of images with length equal to num_frames.
    #     for j in range(num_frames, len(images_predict_batch)):
    #         img = images_predict_batch[j]
    #         if hasattr(img, "convert"):
    #             img = np.array(img)/255.0
    #             # img = np.array(img)
    #         axes[current_row][j-num_frames].imshow(img)
    #         axes[current_row][j-num_frames].axis("off")
    
    # Build the overall title.
    # title = f"Views: {view_indices}\nFrames: {frame_indices}"
    title = 'test'
    if warp_images is not None:
        title += "\n(Warp images appended)"
    if images_predict_batch is not None:
        title += "\n(Predicted images appended)"
    plt.suptitle(title)

    # Save and close the figure.
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# --- Example Usage ---
if __name__ == '__main__':

    #set torch seed as 42
    # Create a generator and set a fixed seed.
    generator = torch.Generator()
    generator.manual_seed(42)

    random.seed(42)

    root = "/dataset/yyzhao/Sync4D"  # Base directory

    # Create datasets for train split
    dataset_syncam_train = Syncam4DDataset(root, split="train")
    dataset_kubric_train = Kubric4DDataset(root, split="train")
    dataset_obj4d_train = Obj4D10kDataset(root, split="train")

    # Create DataLoaders for each training set
    loader_syncam = DataLoader(dataset_syncam_train, batch_size=1, shuffle=True)
    loader_kubric = DataLoader(dataset_kubric_train, batch_size=1, shuffle=True)
    loader_obj4d = DataLoader(dataset_obj4d_train, batch_size=1, shuffle=True, generator=generator)

    # # Visualize one sample from syncam4d training set
    sample_syncam = next(iter(loader_syncam))
    # sample_kubric = next(iter(loader_kubric))
    # sample_obj4d = next(iter(loader_obj4d))

    # sample_obj4d = loader_obj4d[10]
    print("Syncam4D Train Sample:")
    # visualize_sample(sample_obj4d)

    # # Create combined training dataset
    # combined_train = get_combined_dataset(root, split="train")
    # combined_loader = DataLoader(combined_train, batch_size=1, shuffle=True)

    # # Visualize one sample from combined training set
    # sample_combined = next(iter(combined_loader))
    # print("Combined Train Sample:")
    # visualize_sample(sample_combined)