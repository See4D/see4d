import os
import imageio
from glob import glob

import argparse
import ipdb
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--dir1", type=str, required=True)
parser.add_argument("--dir2", type=str, required=True)
args = parser.parse_args()

if __name__ == '__main__':
    # ipdb.set_trace()
    args.dir1 = args.dir1[:-1] if args.dir1[-1] == '/' else args.dir1
    save_dir = os.path.join(args.data, args.dir1 + "-vs-" + args.dir2)
    vid_paths = sorted(glob(os.path.join(args.data, args.dir1, "*.mp4")))
    vid_paths2 = sorted(glob(os.path.join(args.data, args.dir2, "*.mp4")))
    os.makedirs(save_dir, exist_ok=True)
    # vid_paths2 = sorted(glob(os.path.join(args.data.replace(os.path.basename(args.data), args.dir), "*.mp4")))
    for vp, vp2 in zip(vid_paths, vid_paths2):
        vid1 = np.stack(imageio.mimread(vp), axis=0)
        vid2 = np.stack(imageio.mimread(vp2), axis=0)
        vid = np.concatenate([vid1, vid2], axis=-2)
        save_path = os.path.join(save_dir, os.path.basename(vp))
        
        imageio.mimwrite(save_path, vid, fps=8)
