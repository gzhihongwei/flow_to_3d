import argparse
from config.parser import parse_args
import numpy as np
import cv2
import open3d as o3d
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from flow_utils import read_video, load_flow_model, predict_flow_images, create_flow_mask, correspondences_from_flow_mask, visualized_3d_2frames, visualized_3d_3frames
from views3 import reconstruct_3_frames, invert_similarity_transform, homogenize

def main(args):

    video = read_video(args)[:6]
    print("")

    flow_model = load_flow_model(args)

    # Assuming known intrinsics
    K = np.load("assets/sim_intrinsics.npy").astype(np.float32)
    # K /= 2.08
    # K[:2, 2] = K[:2, 2][::-1]
    # K[:2, 2] = 1024.
    # K[-1, -1] = 1.

    frame_inds = range(0, len(video) - 4, 2)

    X_all = []
    # T_all = [np.eye(4)]
    R0 = np.eye(3)
    t0 = np.zeros((3, 1))
    R1 = None
    t1 = None
    rgb_all = []

    camera_all = [(R0, t0)]

    for i in frame_inds:
        X, R1_, R2, t1_, t2, rgb = reconstruct_3_frames(K, R0, t0, video, i, flow_model, args, R1, t1)
        # T = np.eye(4)
        # T[:3, :3] = R1_
        # T[:3, 3] = t1_
        X_all.append(X)
        rgb_all.append(rgb)
        camera_all.append((R1_, t1_))
        R0, t0 = R1_.copy(), t1_.copy()
        R1, t1 = R2.copy(), t2.copy()
        # T_all.append(T @ T_all[-1])
        # R1_all.append(R1)
        # t1_all.append(t1)
    camera_all.append((R1, t1))

    # X_canonicalized = [X_all[0]]
    # # R1_applied = [np.eye(3)]
    # # t1_applied = [np.zeros(3)]
    # for i in range(len(X_all)):
    #     # R1_applied.append(R1_all[i].T @ R1_applied[-1])
    #     # t1_applied.append(-t1_all[i] + t1_applied[-1])
    #     # X_canonicalized.append(R1_applied[-1] @ X_all[i] + t1_applied[-1])
    #     X_can = (invert_similarity_transform(T_all[i]) @ homogenize(X_all[i]).T).T
    #     X_can = X_can / X_can[:, np.newaxis, -1]
    #     X_can = X_can[:, :-1]
    #     X_canonicalized.append(X_can)

    # Visualize:
    geometries = visualized_3d_3frames(np.concatenate(X_all, axis=0), camera_all, np.concatenate(rgb_all, axis=0))
    o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

    print("")

    # Input into Global Optimization

    # Get cameras

    # Figure out object motion?

    # TODO:
    # Automate chirality: i.e. count how many triangulated points are in front of all 3 cameras
    # rescale intrinsics: divide everything by ~2.08 (f, cx, cy)
    # 1. Incorporate flow and do bundle adjustment for just 3 frames of flow at first
    # Remember to multiply flow masks for i+1 and i+2
    #    Which might need incremental SfM so that we can keep optimizing properly?
    #       How do we initialize the extrinsics properly?
    # 2. Consider what quantitative metrics we should be showing?
    # 3. Do we have anything to compare to?

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')
    parser.add_argument('--video_path', help='relative file path of the video', type=str, default=None)
    parser.add_argument('--thresh', help='relative file path of the video', type=float, default=None)
    parser.add_argument('--batch', help='relative file path of the video', type=int, default=None)
    parser.add_argument('--skip', help='relative file path of the video', type=int, default=None)
    args = parse_args(parser)
    
    main(args)