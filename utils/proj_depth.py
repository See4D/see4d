import argparse
import fnmatch
import os
import os.path as osp
from glob import glob
from typing import Literal

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Pipeline, pipeline
import open3d as o3d
import json
from multiprocessing import Pool, cpu_count

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UINT16_MAX = 65535



def project_aligned_monodepth(
    sparse_dir: str,
    img_dir: str,
    mask_dir: str | None,
    save_dir: str,
    aligned_monodepth_dir: str,
    matching_pattern: str = "*",
):
    from pycolmap import SceneManager

    manager = SceneManager(sparse_dir)
    manager.load()

    cameras = manager.cameras
    images = manager.images
    points3D = manager.points3D
    point3D_id_to_point3D_idx = manager.point3D_id_to_point3D_idx

    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

    images = [
        image
        for _, image in images.items()
        if fnmatch.fnmatch(image.name, matching_pattern)
    ]
    all_points3D = []
    # all_rgbs = []
    # for idx, image in enumerate(tqdm(images, "Projecting pixels to point cloud with aligned depth")):
    for idx, image in enumerate(images):
        rgbs = iio.imread(osp.join(img_dir, image.name))
        rgbs = rgbs.reshape(-1, 3)
        if mask_dir:
            mask = iio.imread(osp.join(mask_dir, image.name[:-4] + ".png")) > 0
            if len(mask.shape) == 3:
                mask = mask.sum(-1).astype(bool)
            mask = mask.flatten()
        
        point3D_ids = image.point3D_ids
        point3D_ids = point3D_ids[point3D_ids != manager.INVALID_POINT3D]
        pts3d_valid = points3D[[point3D_id_to_point3D_idx[id] for id in point3D_ids]]  # type: ignore
        K = cameras[image.camera_id].get_camera_matrix()
        rot = image.R()
        trans = image.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        extrinsics = np.linalg.inv(w2c) # c2w
        # import ipdb; ipdb.set_trace()
        disp = np.load(
            osp.join(aligned_monodepth_dir, image.name.split(".")[0] + ".npy"),
        ).flatten()
        valid = disp > 0
        if mask_dir:
            valid = valid & mask
        depth = 1.0 / disp[valid]
        rgbs = rgbs[valid]
        # print(depth.shape)
        
        # Generate grid of (u, v) coordinates
        u, v = np.meshgrid(np.arange(cameras[image.camera_id].width), np.arange(cameras[image.camera_id].height))
        fx, fy ,cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        # Convert image coordinates to normalized camera coordinates
        # import ipdb; ipdb.set_trace()
        x = (u - cx) / fx
        y = (v - cy) / fy
        x = x.flatten()[valid]
        y = y.flatten()[valid]
        Z = depth

        points_cam = np.stack([x*Z, y*Z, Z], axis=-1) # N,3
        R = extrinsics[:3, :3] # 3,3
        t = extrinsics[:3, 3:] # 3,1
        points_world = R[None] @ points_cam[...,None] + t[None] # 3,1
        pts3d_world = points_world[...,0]
        
        all_points3D.append(pts3d_world)
        # all_rgbs.append(rgbs)
        
        ## project the points to near by frame
        if idx < len(images) - 1:
            nextframe_name = images[idx+1].name
            # Transform the 3D points to the camera coordinate system
            R = images[idx+1].R()
            t = images[idx+1].tvec.reshape(3, 1)
            pts3d_cam = (R @ pts3d_world.T + t).T

            # Project the 3D points to the 2D image plane
            K = cameras[images[idx+1].camera_id].get_camera_matrix()
            pts2d_homo = (K @ pts3d_cam.T).T
            pts2d = pts2d_homo[:, :2] / pts2d_homo[:, 2:]

            # Filter points that are within the image bounds
            height, width = cameras[images[idx+1].camera_id].height, cameras[images[idx+1].camera_id].width # Define the size of the output image
            valid_idx = (pts2d[:, 0] >= 0) & (pts2d[:, 0] < width) & (pts2d[:, 1] >= 0) & (pts2d[:, 1] < height)

            # Get valid points and colors
            valid_pts2d = pts2d[valid_idx]
            valid_colors = rgbs[valid_idx]
            ## red denotes the frist mask
            valid_mask_color = np.zeros_like(valid_colors)
            valid_mask_color[:,0] = 255

            # Convert to integer pixel coordinates
            valid_pts2d = valid_pts2d.astype(int)

            # Create an empty image
            image = np.zeros((height, width, 3), dtype=np.uint8)

            # Draw the points on the image using advanced indexing
            image[valid_pts2d[:, 1], valid_pts2d[:, 0]] = valid_colors
            projected_mask = np.zeros((height, width, 3), dtype=np.uint8)
            projected_mask[valid_pts2d[:, 1], valid_pts2d[:, 0]] = valid_mask_color
            save_path = os.path.join(save_dir, "projected-frame", nextframe_name)
            if mask_dir:
                save_path = os.path.join(save_dir, "projected-mask-frame", nextframe_name)
                nextframe = iio.imread(osp.join(img_dir, nextframe_name))
                nextmask = iio.imread(osp.join(mask_dir, nextframe_name[:-4] + ".png")) > 0
                if len(nextmask.shape) == 3:
                    # h,w,3, bool
                    nextmask = nextmask.sum(-1).astype(bool)
                if nextmask.sum() < 10:
                    ## object is missing in this frame, not worth saving
                    break
                
                next_mask_frame = nextframe * nextmask[..., None]
                save_next_mask_frame_path = save_path.replace("projected-mask-frame", "masked-next-frame")
                os.makedirs(os.path.dirname(save_next_mask_frame_path), exist_ok=True)
                iio.imwrite(save_next_mask_frame_path, next_mask_frame) ## masked next frame
                
                fused_frame = next_mask_frame * 0.5 + image * 0.5
                fused_frame_path = save_path.replace("projected-mask-frame", "fused-frame")
                os.makedirs(os.path.dirname(fused_frame_path), exist_ok=True)
                iio.imwrite(fused_frame_path, fused_frame.astype(np.uint8)) # fuse the rgb of projected frame and next frame
                
                nextmask_color = nextmask[..., None] * np.array([0, 255, 0]) # H,W,3
                fused_mask = np.clip(nextmask_color + projected_mask, 0, 255)
                fused_mask_path = save_path.replace("projected-mask-frame", "fused-mask")
                os.makedirs(os.path.dirname(fused_mask_path), exist_ok=True)
                iio.imwrite(fused_mask_path, fused_mask.astype(np.uint8)) # fuse the mask of projected frame and next frame, red for projected, green for next frame
                
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            iio.imwrite(
                save_path, image
            )
    
    return all_points3D


def process_seq(arguments):
    seq, mask_id, args = arguments
    tqdm.write(f"Processing {seq} Mask {mask_id}")
    depth_dir = os.path.join(args.data_root, seq,  "depth")

    out_raw_dir = os.path.join(depth_dir, "raw")
    os.makedirs(out_raw_dir, exist_ok=True)

    out_aligned_dir = os.path.join(depth_dir, "aligned")
    os.makedirs(out_aligned_dir, exist_ok=True)
    
    img_dir = os.path.join(args.data_root, seq,  "images")
    sparse_dir = os.path.join(args.data_root, seq,  args.sparse_dir, "0") 
    mask_dir = os.path.join(args.data_root, seq,  "vos-results", mask_id)
    save_dir = os.path.join(args.data_root, seq,  "instance_mask_projected", mask_id)

    if not os.path.exists(os.path.join(args.data_root, seq,  f"depth/projected_points_masked.ply")):
        tqdm.write(f"{seq} do not have valid depth points, skipping")
        return

    try:
        all_points3D = project_aligned_monodepth(
            sparse_dir,
            img_dir,
            mask_dir,
            save_dir,
            out_aligned_dir,
            args.matching_pattern,
        )

    except Exception as e:
        tqdm.write(f"Error projecting aligned monodepth for seq -->> {seq} << --: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything-v2",
        help="depth model to use, one of [depth-anything, depth-anything-v2]",
    )
    parser.add_argument("-d","--data_root", type=str, required=True)
    parser.add_argument("--index_path", type=str, default=None, required=True)
    parser.add_argument("--out_raw_dir", type=str, default=None)
    parser.add_argument("--out_aligned_dir", type=str, default=None)
    parser.add_argument("--sparse_dir", type=str, default="sparse")
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--metric_dir", type=str, default=None)
    parser.add_argument("--matching_pattern", type=str, default="*")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--project", action="store_true", help="project points to 3d")
    
    parser.add_argument("-n","--num_workers", type=int, default=32, help="number of parallel workers")
    
    args = parser.parse_args()

    seq_list = json.load(open(args.index_path))
    pool_args = []
    for seq, masks in seq_list.items():
        for mask_id in masks:
            pool_args.append((seq, mask_id, args))

    
    # Use multiprocessing to process the videos
    num_workers = min(args.num_workers, len(seq_list), cpu_count())
    if num_workers > 1:
        with Pool(num_workers) as pool:
            list(tqdm(pool.imap(process_seq, pool_args), total=len(seq_list)))
    else:
        for arg in tqdm(pool_args):
            process_seq(arg)
            
        


if __name__ == "__main__":
    """ example usage for iphone dataset:
    python datatools/depth/project_instance_mask.py -n 16 -d datasets/davis-24 --sparse_dir sfm-mask/model --index_path datasets/davis-24/valid_masks.json
     
    # DAVIS
    # python datatools/depth/project_instance_mask.py -n 16 -d datasets/davis-24 --sparse_dir sfm-mask/model --index_path outputs-svd/DAVIS/manual_filter.json 
    
    # webvid17k part1 v9
    # python datatools/depth/project_instance_mask.py -n 16 -d datasets/webvid/filtered-17K --sparse_dir sfm-mask/model --index_path datasets/4d-data-list/webvid-17k_part_1.json

    # webvid17k part2 v14
    # python datatools/depth/project_instance_mask.py -n 16 -d datasets/webvid/filtered-17K --sparse_dir sfm-mask/model --index_path datasets/4d-data-list/webvid-17k_part_2.json
    
    # openvid-26k v1
    # python datatools/depth/project_instance_mask.py -n 16 -d datasets/openvid-mask-data --sparse_dir sfm-mask/model --index_path datasets/4d-data-list/openvid-26k_part_1.json

    # openvid-26k v2
    # python datatools/depth/project_instance_mask.py -n 16 -d datasets/openvid-mask-data --sparse_dir sfm-mask/model --index_path datasets/4d-data-list/openvid-26k_part_2.json

    # openvid-26k re10k-part2
    # python datatools/depth/project_instance_mask.py -n 16 -d datasets/openvid-mask-data --sparse_dir sfm-mask/model --index_path datasets/4d-data-list/openvid-26k_part_3.json

    # vipseg-1710
    # python datatools/depth/project_instance_mask.py -n 16 -d datasets-azure/VIPSeg/videos --sparse_dir sfm/model --index_path datasets-azure/VIPSeg/videos/valid_masks.json

    # ytvis-1810 v3
    # python datatools/depth/project_instance_mask.py -n 16 -d datasets-azure/Youtube-VIS-Videos/videos --sparse_dir sfm/model --index_path datasets-azure/Youtube-VIS-Videos/videos/valid_masks.json
    """
    main()