import os
import torch, os, imageio
from torchvision.transforms import v2
from einops import rearrange
import torchvision
from PIL import Image
import numpy as np
import random
from pathlib import Path
from utils.train_utils import mask_pixels
from decord import VideoReader
import torch.nn.functional as F
from utils.train_utils import random_edge_mask

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split, num_frames_sample, frame_interval=4, target_resolution=None):
        self.data_root = data_root
        self.split = split
        self.frame_interval = frame_interval
        self.target_resolution = target_resolution
        self.num_frames_sample = num_frames_sample
        self.max_num_frames = 81
            
        # self.frame_process = v2.Compose([
        #     # v2.CenterCrop(size=(height, width)),
        #     v2.Resize(size=(height, width), antialias=True),
        #     v2.ToTensor(),
        #     v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # ])

        self.prepare_dataset()
        
    def prepare_dataset(self):
        #read all paths
        path_list = []
        cam_cats = os.listdir(Path(self.data_root) / self.split)
        for _cam in cam_cats:
            cam_path = Path(self.data_root) / self.split / _cam
            path_list.extend([str(cam_path) + f'/{scene_name}' for scene_name in  os.listdir(cam_path)])
        self.path = path_list

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)

        #read data, 81 frames, 以最大interval=4,随机选一个interval,然后选取8帧，然后转成torch，插值到512*512
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        return frames


    def load_video(self, video_path, frames):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, 
                        start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def load_video_decord(self, video_path, frame_indices):
        # ctx=cpu(0) 表示用 CPU 读取
        vr = VideoReader(video_path)
        # frame_indices: list 或 1D Tensor，里面是要取的帧号
        # get_batch 会一次性解码所有这些帧
        batch = vr.get_batch(list(frame_indices)).asnumpy()  # (T, H, W, C)
        video = torch.from_numpy(batch).float() / 255.0  # now in [0,1]

        video = video.permute(0,3,1,2)

        # processed = []
        # for img_np in batch:
        #     img = Image.fromarray(img_np)
        #     img = self.crop_and_resize(img)
        #     processed.append(frame_process(img))

        # # stack 并调整成 (C, T, H, W)
        # frames = torch.stack(processed, dim=0)
        return video

    def __getitem__(self, idx):

        path = os.path.join(self.path[idx], 'videos')
        video_path_list = sorted(os.listdir(path))
        source_video_path = os.path.join(path, video_path_list[random.randrange(0,9)])
        target_video_path = os.path.join(path, video_path_list[-1])

        interval = random.randint(1, self.frame_interval)

        max_start = self.max_num_frames - interval * (self.num_frames_sample - 1)
        start = random.randrange(max_start)
        frames = torch.arange(self.num_frames_sample, dtype=torch.long) * interval + start

        source_video = self.load_video_decord(source_video_path, frames)
        target_video = self.load_video_decord(target_video_path, frames)

        f, c, h, w = source_video.shape

        if h != self.target_resolution[0] or w != self.target_resolution[1]:
            source_video = F.interpolate(source_video, size=self.target_resolution, mode="bilinear", align_corners=False)
            target_video = F.interpolate(target_video, size=self.target_resolution, mode="bilinear", align_corners=False)

        video_sample = torch.concat([source_video, target_video], dim=0)
        _, target_mask = mask_pixels(target_video)
        target_mask = random_edge_mask(target_mask)

        return {
            "video": video_sample,
            "target_mask": target_mask,
        }

    
    def __len__(self):
        return len(self.path)

if __name__ == '__main__':
    dataset = TextVideoDataset(
        data_root='data/MultiCamVideo-Dataset',
        split='train',
        frame_interval=10,
        height=480, 
        width=832
    )
    item = dataset.__getitem__(0)
