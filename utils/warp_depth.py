import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def unproject_and_warp(
    img_dir: str,
    disparity_dir: str,
    save_dir: str,
    intrinsic_matrix: np.ndarray
):
    """
    Unproject images into 3D, and transform/warp into nearby rotated frames.

    Args:
        img_dir (str): Directory containing input .png images.
        disparity_dir (str): Directory containing .npy disparity maps for each frame.
        save_dir (str): Directory to save the warped frames.
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix (3x3).
        rotation_angles (list[float]): List of rotation angles in degrees around the y-axis in the camera frame.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load all image filenames
    #img_filenames = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    img_filenames = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")], key = lambda x: int(x.split('.')[0].split('_')[-1]))
    disparities = np.load(disparity_dir)

    # Define rotation angles
    rotation_angles = np.linspace(-10, 10, 15).tolist()

    # Define rotation center based on the first frame
    img_path = os.path.join(img_dir, img_filenames[0])
    img = cv2.imread(img_path)
    depth = 1.0 / (disparities[0] + 1e-6)
    valid_mask = disparities[0] > 0
    
    height, width = img.shape[:2]
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - cx) / fx
    y = (v - cy) / fy

    Z = depth[valid_mask]
    X = x[valid_mask] * Z
    Y = y[valid_mask] * Z
    points_3D = np.stack([X, Y, Z], axis=-1)

    median_depth_idx = np.argsort(Z)[len(Z) // 5]
    rotation_center = points_3D[median_depth_idx]

    for idx, img_name in enumerate(img_filenames):

        frame_dir = os.path.join(save_dir, f"frame_{idx}")
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        # Load image and disparity
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        depth = 1.0 / (disparities[idx] + 1e-6)
        valid_mask = disparities[idx] > 0

        #save img
        reference_img_dir = os.path.join(frame_dir, 'reference_images')
        if not os.path.exists(reference_img_dir):
            os.makedirs(reference_img_dir)
        cv2.imwrite(os.path.join(reference_img_dir, img_name), img)

        # Unproject to 3D
        Z = depth[valid_mask]
        X = x[valid_mask] * Z
        Y = y[valid_mask] * Z
        points_3D = np.stack([X, Y, Z], axis=-1)

        for angle in rotation_angles:
            warp_img_dir = os.path.join(frame_dir, 'warp_images')
            if not os.path.exists(warp_img_dir):
                os.makedirs(warp_img_dir)

            # Define rotation matrix around the y-axis in the camera frame
            rot_mat = R.from_euler('y', angle, degrees=True).as_matrix()

            # Apply rotation around the rotation center
            translated_points = points_3D - rotation_center
            rotated_points = (rot_mat @ translated_points.T).T + rotation_center

            # Project back to 2D for the new frame
            projected_points = (intrinsic_matrix @ rotated_points.T).T
            projected_points /= projected_points[:, 2:3]  # Normalize by depth

            # Create warped image
            warped_img = np.zeros_like(img)
            valid_proj = (
                (projected_points[:, 0] >= 0) & (projected_points[:, 0] < width) &
                (projected_points[:, 1] >= 0) & (projected_points[:, 1] < height)
            )
            valid_uv = projected_points[valid_proj, :2].astype(int)
            warped_img[valid_uv[:, 1], valid_uv[:, 0]] = img[valid_mask][valid_proj]

            mask = np.zeros_like(img)
            mask[valid_uv[:, 1], valid_uv[:, 0]] = 1
            
            # Save warped image
            save_path = os.path.join(warp_img_dir, f"warp_{img_name.replace('.png', '')}_rot{angle:+03f}.png")
            cv2.imwrite(save_path, warped_img)
            # save mask
            save_path = os.path.join(warp_img_dir, f"mask_{img_name.replace('.png', '')}_rot{angle:+03f}.png")
            cv2.imwrite(save_path, (mask*255).astype(np.uint8))

# Example usage
intrinsic_matrix = np.array([
    [800, 0, 1280],
    [0, 800, 704],
    [0, 0, 1],
])  # Example intrinsic matrix for a 640x352 camera
unproject_and_warp(
    img_dir="/dataset/htx/see4d/inputs/images2",
    disparity_dir="/dataset/htx/see4d/depths/purse.npy",
    save_dir="/dataset/htx/see4d/warps/outputs/purse",
    intrinsic_matrix=intrinsic_matrix
)
#unproject_and_warp(
#    img_dir="/data/dylu/project/DepthAnyVideo/outputs/img",
#    disparity_dir="/data/dylu/project/DepthAnyVideo/outputs/cat2.npy",
#    save_dir="/data/dylu/project/DepthAnyVideo/outputs/cat2",
#    intrinsic_matrix=intrinsic_matrix
#)
