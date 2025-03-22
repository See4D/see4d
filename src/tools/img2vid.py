import os
import imageio
from glob import glob

import argparse
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--save_name", type=str, default="output.mp4")
args = parser.parse_args()

if __name__ == '__main__':
    all_data_dir =  sorted(glob(os.path.join(args.data, "*")))
    print(all_data_dir)
    # ipdb.set_trace()
    for data_dir in all_data_dir:
        if os.path.isdir(data_dir):
            image_paths = sorted(glob(os.path.join(data_dir, "**/*.png")))
            if len(image_paths) == 0:
                continue
            images = [imageio.imread(im) for im in image_paths]
            save_path = os.path.join(os.path.dirname(data_dir), os.path.basename(data_dir)+".mp4")
            # ipdb.set_trace()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.mimwrite(save_path, images, fps=8)