
import numpy as np
import torch
import json

from diffusion.utils.cam_utils import get_interpolated_poses

context_rt1 = [
                [
                    9.9829e-01,
                    1.7920e-03,
                    -5.8455e-02,
                    -1.9491e+00
                ],
                [
                    -8.6971e-03,
                    9.9297e-01,
                    -1.1809e-01,
                    3.4287e-01
                ],
                [
                    5.7832e-02,
                    1.1839e-01,
                    9.9128e-01,
                    -4.2531e+00
                ]
            ]

context_rt2 = [
                [
                    0.9946,
                    0.0176,
                    -0.1025,
                    2.6288
                ],
                [
                    -0.0163,
                    0.9998,
                    0.0136,
                    0.7943
                ],
                [
                    0.1027,
                    -0.0119,
                    0.9946,
                    -2.0061
                ]
            ]

context_rt1 = np.array(context_rt1)
context_rt2 = np.array(context_rt2)

new_poses = get_interpolated_poses(context_rt1, context_rt2) # list
pose_json = [
    {"pose": pp[:3].tolist()} for pp in new_poses
]

with open("poses.json", "w") as f:
    json.dump(pose_json, f, indent=4)

