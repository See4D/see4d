import os
import cv2
import sys

def save_frames(video_path, interval):
    # Create a directory with the same name as the video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(video_path), video_name)
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read and save each frame
    frame_count = 0
    save_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % int(interval) != 0:
            frame_count += 1
            continue
        else:
            save_index += 1
            frame_count += 1
        # Save the frame as an image file
        frame_path = os.path.join(output_dir, f"{save_index:03d}.jpg")
        cv2.imwrite(frame_path, frame)

    # Release the video capture object
    cap.release()

    print(f"Frames saved to: {output_dir}")

if __name__ == "__main__":
    # Read the video path from the command line
    video_path = sys.argv[1]
    interval = sys.argv[2]

    # Call the function to save frames
    save_frames(video_path, interval)