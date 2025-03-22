import os
import cv2

def create_viewpoint_videos(frame_dir: str, output_dir: str):
    """
    Create videos for each viewpoint by collecting images with similar rotation identifiers across all frame directories.

    Args:
        frame_dir (str): Directory containing frame directories (e.g., frame_0, frame_1, ...).
        output_dir (str): Directory to save the output videos for each viewpoint.
    """

    # Get the list of frame directories
    # frame_dirs = sorted([d for d in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, d))])
    frame_dirs = [f"frame_{i}" for i in range(27)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not frame_dirs:
        raise ValueError(f"No frame directories found in {frame_dir}")

    # Extract viewpoint identifiers from the first frame directory
    first_frame_dir = os.path.join(frame_dir, frame_dirs[0])
    files_in_first_frame = sorted([f for f in os.listdir(first_frame_dir) if f.endswith(".png") or f.endswith(".jpg")])

    print(first_frame_dir)

    # Extract unique rotation identifiers
    viewpoint_identifiers = set()
    for file_name in files_in_first_frame:
        if "rot" in file_name:
            rot_identifier = file_name.split("rot")[1]  # Extract everything after "rot"
            viewpoint_identifiers.add(rot_identifier)

    viewpoint_identifiers = sorted(viewpoint_identifiers)  # Ensure consistent order

    for rot_identifier in viewpoint_identifiers:
        # Prepare video writer for this viewpoint
        output_path = os.path.join(output_dir, f"viewpoint_{rot_identifier.replace('.png', '').replace('.jpg', '')}.mp4")

        # Example frame to get dimensions
        example_frame_path = None
        for frame_dir_name in frame_dirs:
            frame_dir_path = os.path.join(frame_dir, frame_dir_name)
            candidate_frames = [f for f in os.listdir(frame_dir_path) if f.endswith(rot_identifier)]
            if candidate_frames:
                example_frame_path = os.path.join(frame_dir_path, candidate_frames[0])
                break

        if not example_frame_path:
            print(f"No frames found for viewpoint {rot_identifier}, skipping.")
            continue

        frame_example = cv2.imread(example_frame_path)
        height, width, _ = frame_example.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, 2, (width, height))

        for frame_dir_name in frame_dirs:
            frame_dir_path = os.path.join(frame_dir, frame_dir_name)
            candidate_frames = [f for f in os.listdir(frame_dir_path) if f.endswith(rot_identifier)]

            if candidate_frames:
                frame_path = os.path.join(frame_dir_path, candidate_frames[0])
                frame = cv2.imread(frame_path)
                video_writer.write(frame)
            else:
                print(f"Warning: No frame for {rot_identifier} in {frame_dir_path}")

        video_writer.release()
        print(f"Video saved: {output_path}")


# Example usage
create_viewpoint_videos("/dataset/htx/see4d/outputs/cat_reverse_k3", "/dataset/htx/see4d/outputs/cat_nofilterk3_video")
