import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from config.parser import parse_args
from flow_utils import read_video, load_flow_model, visualized_3d_3frames
from views import reconstruct_3_frames


def main(args):
    video = read_video(args)[10:22]  # If you only want a certain subset of frames

    flow_model = load_flow_model(args)

    # Assuming known intrinsics
    K = np.load(args.intrinsics_path).astype(np.float32)
    # Intrinsics were taken from Assignment 3, but the orientation was portrait and the aspect ratio was different. Uncomment to get fixed intrinsics
    # K /= 2.08
    # K[:2, 2] = K[:2, 2][::-1]
    # K[-1, -1] = 1.

    frame_inds = range(0, len(video) - 2 * args.skip, args.skip)

    X_all = []
    R0 = np.eye(3)
    t0 = np.zeros((3, 1))
    R1 = None
    t1 = None
    rgb_all = []

    camera_all = [(R0, t0)]

    for i in frame_inds:
        X, R1_, R2, t1_, t2, rgb = reconstruct_3_frames(K, R0, t0, video, i, flow_model, args, R1, t1)
        X_all.append(X)
        rgb_all.append(rgb)
        camera_all.append((R1_, t1_))
        R0, t0 = R1_.copy(), t1_.copy()
        R1, t1 = R2.copy(), t2.copy()
        ## Uncomment if you want to progressively view the frames and 3D reconstruction
        # geometries = visualized_3d_3frames(np.concatenate(X_all, axis=0), camera_all, np.concatenate(rgb_all, axis=0))
        # o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")
    camera_all.append((R1, t1))

    # Visualize all the frames and the 3D reconstruction
    geometries = visualized_3d_3frames(np.concatenate(X_all, axis=0), camera_all, np.concatenate(rgb_all, axis=0))
    o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', default=Path(__file__).resolve().parent / "config/eval/spring-M.json", type=Path)
    parser.add_argument('--url', help='checkpoint url', type=str, default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M")
    parser.add_argument('--device', help='inference device', type=str, default='cuda')
    parser.add_argument('--intrinsics_path', help='path to intrinsics', type=str, required=True)
    parser.add_argument('--video_path', help='relative file path of the video', type=str, required=True)
    parser.add_argument('--thresh', help='The threshold for the flow to segment the moving object', type=float, required=True)
    parser.add_argument('--skip', help='How many frames to skip', type=int, required=True)
    args = parse_args(parser)
    
    main(args)