import open3d as o3d
import numpy as np
import torch
import decord
decord.bridge.set_bridge("torch")
import argparse
from config.parser import parse_args
import torch.nn.functional as F
import cv2

from decord import VideoReader

from sea_raft.raft import RAFT
from views import Views, get_correspondences, ransac_F, F_to_E, F_to_P, triangulate, poses_from_E, visualize_poses, plot_epipolar_lines

from tqdm import tqdm


@torch.no_grad()
def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1.to(torch.float32), scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2.to(torch.float32), scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

def read_video_decord(video_path: str):
  vr = VideoReader(video_path)
  frames = vr.get_batch(list(range(len(vr))))
  return frames

def optimize(args, video, model):

    start = 20
    end = 24

    # tensor = torch.tensor([1]).cuda()
    video = read_video_decord(video).permute(0, 3, 1, 2).to(args.device)
    # TODO: might need to downsample/resize down
    image1 = video[start : end-3]
    image2 = video[start+3 : end]
    flow, _ = calc_flow(args, model, image1, image2)
    flow = flow.permute(0, 2, 3, 1)

    # pts1, pts2, rgb = get_correspondences(flow, image1)
    # pts1 = pts1.cpu().detach().numpy()
    # pts2 = pts2.cpu().detach().numpy()

    pts1 = np.load("sanity/pts1.npy")
    pts2 = np.load("sanity/pts2.npy")
    P1_true = np.load("sanity/P1.npy")
    P2_true = np.load("sanity/P2.npy")
    img1 = cv2.imread("sanity/img1.jpg")
    img2 = cv2.imread("sanity/img2.jpg")
    rgb = None

    pts1_norm = pts1 - np.array([img1.shape[1]/2, img1.shape[0]/2])
    pts2_norm = pts2 - np.array([img1.shape[1]/2, img1.shape[0]/2])
    
    F, inliers = ransac_F(pts1, pts2, thresh = 1.)
    print("Inlier ratio = ", inliers)
    inds = np.arange(8) * 1000
    # plot_epipolar_lines(F, pts1_norm[inds], pts2_norm[inds], img1, img2)
    # geometries = visualize_poses(P1_true, P2_true, pts1, pts2, rgb)
    # o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

    P = F_to_P(F, v = np.array([0., 0., 0.]), lamb = 1.)
    #array([ 9.99940307e-01,  1.09262649e-02, -1.06444347e-06])

    geometries = visualize_poses(P, pts1, pts2, rgb)
    o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

    # E = F_to_E(F)
    poses = poses_from_E(F)

    P = np.zeros((3, 4))
    for pose in poses:

        P[:, :3] = pose[1] # R
        P[:, 3] = pose[0] #pose[1] @ -pose[0]  # C

        X = triangulate(P, pts1, pts2)

        # cheirality = (X - pose[0].cpu()) > 0  # TODO: premultiply row 3

        # print(cheirality.sum())
        geometries = visualize_poses(P, pts1, pts2, rgb)
        o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

    views = Views(args, flow)
    views = views.to(args.device)

    # TODO:
    # 1. Plot the initial estimate of the camera poses in open3d
    # 2. Initialize camera poses better (solving 8-pt algorithm?)
    # 3. Take a slow motion video

    geometries = views.visualize(video[start : end])
    o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

    for _ in tqdm(range(args.iterations)):
        views.optimization_step()
    print(views.rl)#, views.pl)
    # print(views.rots)
    # print(views.trans)
    # print(views.focal)

    # Visualize the point cloud
    pts_vis = views.visualize(video[start : end])
    o3d.visualization.draw_geometries(pts_vis, window_name="Random Point Cloud")

    # Red: X-axis
    # Green: Y-axis
    # Blue: Z-axis

    print("")


def is_module_on_device(module, device_type="cuda"):
    for name, param in module.named_parameters():
        if param.device.type != device_type:
            print(name)
            return False
    for name, buffer in module.named_buffers():
        if buffer.device.type != device_type:
            print(name)
            return False
    for key, value in module.__dict__.items():
        if isinstance(value, torch.Tensor) and value.device.type != device_type:
            print(key)
            return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')
    args = parse_args(parser)

    model = RAFT.from_pretrained(args.url, args=args)
        
    # if args.device == 'cuda':
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')
    model = model.to(args.device)
    # print(device)
    model.eval()
    # print("model moved")
    # demo_custom(model, args, device=device)
    # flow_video(model, args, device=device)
    # track_rigid_body(model, args, "assets/videos/rubiks_cube.mp4", device)
    # tensor = torch.tensor([1]).to(device)
    # video = read_video_decord(video).permute(0, 3, 1, 2).to(device)
    # image1 = video[:-1]
    # image2 = video[1:]
    # flow, _ = calc_flow(args, model, image1, image2)
    # views = Views(args, flow.cpu())
    # views = views.to(args.device)
    # views.optimization_step()
    optimize(args, "assets/videos/rubiks_cube.mp4", model)

if __name__ == '__main__':
    main()