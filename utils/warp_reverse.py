import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import shutil
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter, uniform_filter, map_coordinates
from scipy.spatial import KDTree
from joblib import Parallel, delayed

def inpaint_cv_depth(depth_map, method='telea', inpaint_radius=3):
    mask = (depth_map == 0).astype(np.uint8)

    if method == 'telea':
        flags = cv2.INPAINT_TELEA
    elif method == 'ns':
        flags = cv2.INPAINT_NS
    else:
        raise ValueError("Invalid method. Use 'telea' or 'ns'.")

    dense_depth = cv2.inpaint(depth_map.astype(np.float32), mask, inpaint_radius, flags)

    return dense_depth

def inpaint_inter_depth(depth_map):
    y, x = np.where(depth_map > 0)
    values = depth_map[y, x]

    grid_x, grid_y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))

    dense_depth = griddata((x, y), values, (grid_x, grid_y), method='linear')

    dense_depth[np.isnan(dense_depth)] = 0  #

    return dense_depth

def replace_with_nearest(depth_map, k=3):
    """
    Replace missing depth (0 values) with the nearest valid depth within a k*k window.
    
    Parameters:
    - depth_map: numpy.ndarray, sparse depth map with 0 indicating missing values.
    - k: int, size of the neighborhood window (must be odd).
    
    Returns:
    - inpainted_depth: numpy.ndarray, depth map with missing values filled.
    """
    def nearest_valid_depth(values):
        center = len(values) // 2
        #if values[center] > 0:  # Already valid, no change
        #    return values[center]
        valid_values = values[values > 0]
        return min(valid_values) if len(valid_values) > 0 else 0  # Replace with nearest non-zero depth
    
    # Apply a filter to find nearest valid depth
    inpainted_depth = generic_filter(
        depth_map,
        function=nearest_valid_depth,
        size=k,
        mode='constant',
        cval=0
    )
    return inpainted_depth

def detect_and_mask_outliers(mask0, m=3, weights = 0.2):
    # Step 1: Generate initial mask
    mask = (mask0 > 0).astype(np.float32)
    #print(mask.shape)
    #print(depth_map.min(), depth_map.max())

    # Step 2: Compute the local sum of the mask using a uniform filter
    local_sum = uniform_filter(mask, size=m, mode="constant")

    # Step 3: Compute the average of all local sums
    average_local_sum = np.mean(local_sum)

    # Step 4: Mark outliers
    updated_mask = np.ones_like(mask.copy())
    updated_mask[local_sum < weights * average_local_sum] = 0

    return updated_mask

def warp_back_to_original(
    updated_warped_depth, img, intrinsic_matrix, rotation_center, rotation_angle, width, height
):
    """
    Warp the updated depth back to the original image using bilinear interpolation.

    Parameters:
    - updated_warped_depth: numpy.ndarray, updated warped depth map.
    - img: numpy.ndarray, original image.
    - intrinsic_matrix: numpy.ndarray, 3x3 intrinsic camera matrix.
    - rotation_center: numpy.ndarray, center of rotation in 3D space.
    - rotation_angle: float, rotation angle in degrees around the y-axis.
    - width: int, width of the image.
    - height: int, height of the image.

    Returns:
    - new_image: numpy.ndarray, warped image back to the original view.
    """
    updated_warped_depth = updated_warped_depth.squeeze(-1)
    # Step 1: Create pixel coordinates
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    valid_mask = updated_warped_depth > 0

    # Step 2: Unproject to 3D
    Z = updated_warped_depth[valid_mask]
    #print(x.shape, valid_mask.shape)
    X = (x[valid_mask] - intrinsic_matrix[0, 2]) / intrinsic_matrix[0, 0] * Z
    Y = (y[valid_mask] - intrinsic_matrix[1, 2]) / intrinsic_matrix[1, 1] * Z
    points_3D = np.stack([X, Y, Z], axis=-1)

    # Step 3: Apply inverse rotation
    rot_mat = R.from_euler('y', -rotation_angle, degrees=True).as_matrix()  # Inverse rotation
    translated_points = points_3D - rotation_center
    original_points = (rot_mat @ translated_points.T).T + rotation_center

    # Step 4: Project back to the original 2D image plane
    projected_points = (intrinsic_matrix @ original_points.T).T
    projected_points /= projected_points[:, 2:3]  # Normalize by depth

    # Step 5: Bilinear interpolation for color
    projected_x = projected_points[:, 0]
    projected_y = projected_points[:, 1]
    valid_proj = (
        (projected_x >= 0) & (projected_x < width) &
        (projected_y >= 0) & (projected_y < height)
    )

    #print(projected_x.shape, valid_proj.shape)

    # Extract valid coordinates
    valid_x = projected_x[valid_proj]
    valid_y = projected_y[valid_proj]
    original_coords = np.stack([valid_y, valid_x], axis=0)

    #print(original_coords.shape, img.shape)

    # Bilinear interpolation on the original image
    #interpolated_colors = map_coordinates(img, original_coords, order=1, mode="nearest")
    if img.ndim == 3 and img.shape[2] == 3:  # RGB Image
        interpolated_colors = np.zeros((len(valid_x), 3), dtype=img.dtype)
        for i in range(3):  # Process each channel
            interpolated_colors[:, i] = map_coordinates(img[:, :, i], original_coords, order=1, mode="nearest")
    else:  # Grayscale Image
        interpolated_colors = map_coordinates(img, original_coords, order=1, mode="nearest")


    # Step 6: Create the new warped image
    new_image = np.zeros_like(img)
    target_coords = np.stack([y[valid_mask][valid_proj], x[valid_mask][valid_proj]], axis=-1)
    #print(target_coords.shape, interpolated_colors.shape)
    #new_image[target_coords[:, 0], target_coords[:, 1], :] = interpolated_colors
    #print(x[valid_mask][valid_proj].max())
    #print(new_image[target_coords[:, 0], target_coords[:, 1]].shape)
    new_image[target_coords[:, 0], target_coords[:, 1]] = interpolated_colors

    return new_image 

def dynamic_radius_denoise(point_cloud, weight=2.0, min_neighbors=10):
    """
    Denoise a point cloud using a dynamic radius (batch processing optimized).
    
    Parameters:
    - point_cloud (numpy.ndarray): Input point cloud, shape (N, 3).
    - weight (float): Weight to scale the dynamic radius. Default is 2.0.
    - min_neighbors (int): Minimum number of neighbors required in the radius. Default is 10.
    
    Returns:
    - filtered_point_cloud (numpy.ndarray): Denoised point cloud.
    """
    # Build a KDTree for the point cloud
    kdtree = KDTree(point_cloud)

    # Batch compute nearest neighbor distances
    distances, _ = kdtree.query(point_cloud, k=2)  # k=2, as the first neighbor is the point itself
    nearest_distances = distances[:, 1]

    # Compute the dynamic radius
    mean_distance = np.mean(nearest_distances)
    radius = mean_distance * weight

    # Filter points based on the dynamic radius
    neighbor_counts = [len(kdtree.query_ball_point(p, radius)) for p in point_cloud]
    indices = [i for i, count in enumerate(neighbor_counts) if count > min_neighbors]

    # Extract the filtered point cloud
    filtered_point_cloud = point_cloud[indices]
    return filtered_point_cloud, indices


def mask_outliers(point_cloud, threshold=1.0):
    """
    Mask out outliers from a point cloud based on nearest neighbor distance.

    Parameters:
    - point_cloud (numpy.ndarray): Input point cloud, shape (N, 3).
    - threshold (float): Maximum allowed nearest neighbor distance. Points with
                         distances above this threshold will be removed.

    Returns:
    - filtered_point_cloud (numpy.ndarray): Point cloud with outliers removed.
    - indices (numpy.ndarray): Indices of the points that are retained.
    """
    # Build a KDTree for nearest neighbor search
    kdtree = KDTree(point_cloud)

    # Compute nearest neighbor distances
    distances, _ = kdtree.query(point_cloud, k=2)  # k=2 to exclude the point itself
    nearest_distances = distances[:, 1]

    # Mask out points with distances greater than the threshold
    mask = nearest_distances <= threshold
    filtered_point_cloud = point_cloud[mask]
    indices = np.where(mask)[0]

    return filtered_point_cloud, indices

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
    if  os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir)

    # Load all image filenames
    #img_filenames = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    img_filenames = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")], key = lambda x: int(x.split('.')[0].split('_')[-1]))
    #print(img_filenames)
    #assert False
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
        ##print(Z.shape, depth.shape, img.shape)
        #assert False
        X = x[valid_mask] * Z
        Y = y[valid_mask] * Z
        points_3D = np.stack([X, Y, Z], axis=-1)
        #print(points_3D.shape)
        #points_3D = remove_outliers_open3d(points_3D, nb_neighbors=20, std_ratio=2.0)
        #print(valid_mask.shape)
        #assert False
        #points_3D, valid_indices = dynamic_radius_denoise(points_3D, weight=1.0, min_neighbors=10)
        #points_3D, valid_indices = mask_outliers(points_3D, 0.1)
        #print(points_3D.shape)
        #assert False

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
            warped_img[valid_uv[:, 1], valid_uv[:, 0]] = img[valid_mask][valid_proj]#valid_indices
            
            warped_depth = np.expand_dims(np.zeros_like(depth), axis=-1)
            warped_depth[valid_uv[:, 1], valid_uv[:, 0]] = np.expand_dims(depth, axis=-1)[valid_mask][valid_proj]#valid_indices

            mask = np.zeros_like(img)[:,:,:1]
            mask[valid_uv[:, 1], valid_uv[:, 0]] = 1
            
            ## Save warped image
            #save_path = os.path.join(warp_img_dir, f"warp_{img_name.replace('.png', '')}_rot{angle:+03f}.png")
            #cv2.imwrite(save_path, warped_img)

            #print(warped_depth.shape)
            #assert False

            save_path = os.path.join(warp_img_dir, f"depth_{img_name.replace('.png', '')}_rot{angle:+03f}.png")
            #warped_depth = inpaint_cv_depth(warped_depth, method='telea', inpaint_radius=3)
            #warped_depth = np.expand_dims(inpaint_inter_depth(warped_depth.squeeze(-1)), axis=-1)
            #mask = detect_and_mask_outliers(mask, m=5, weights=0.3)
            warped_depth = replace_with_nearest(warped_depth, k=3)
            #warped_depth = mask * warped_depth
            #print(warped_depth.shape)
            #warped_depth = detect_and_mask_outliers(warped_depth, m=5, depth_threshold=1e6)
            cv2.imwrite(save_path, (warped_depth*255).astype(np.uint8))

            warped_img = warp_back_to_original(warped_depth, img, intrinsic_matrix, rotation_center, angle, width, height)
            save_path = os.path.join(warp_img_dir, f"warp_{img_name.replace('.png', '')}_rot{angle:+03f}.png")
            cv2.imwrite(save_path, warped_img)

            # save mask
            save_path = os.path.join(warp_img_dir, f"mask_{img_name.replace('.png', '')}_rot{angle:+03f}.png")
            cv2.imwrite(save_path, (mask*255).astype(np.uint8))

            #assert False

# Example usage
intrinsic_matrix = np.array([
    [800, 0, 1280],
    [0, 800, 704],
    [0, 0, 1],
])  # Example intrinsic matrix for a 640x352 camera

unproject_and_warp(
    img_dir="/dataset/htx/see4d/inputs/img",
    disparity_dir="/dataset/htx/see4d/depths/cat.npy",
    save_dir="/dataset/htx/see4d/warps/outputs/cat_reverse_k3",
    intrinsic_matrix=intrinsic_matrix
)
