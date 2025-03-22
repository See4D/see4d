import os
import imageio
import sys
import numpy as np

if __name__ == '__main__':
    vid_path = sys.argv[1]
    num_chunk = int(sys.argv[2]) # 5
    video = imageio.mimread(vid_path)
    save_dir = vid_path[:-4] # remove .mp4
    os.makedirs(save_dir, exist_ok=True)
    videonp = np.stack(video, axis=0) # F,H,NW,3
    video_chunks = np.split(videonp, num_chunk, axis=2)
    for i in range(num_chunk):
        save_vid = list(video_chunks[i])
        imageio.mimwrite(os.path.join(save_dir, f"{i}.mp4"), save_vid)
    
    