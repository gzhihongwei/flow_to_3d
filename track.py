import argparse
import os
import cv2
import math
import numpy as np

import decord
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import copy

from tqdm import tqdm

# import pillow_heif
from decord import VideoReader

from config.parser import parse_args

from sea_raft.raft import RAFT
from sea_raft.utils.flow_viz import flow_to_image

decord.bridge.set_bridge("torch")

def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

@torch.no_grad()
def demo_data(path, args, model, image1, image2):
    os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    flow, info = calc_flow(args, model, image1, image2)
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}flow.jpg", flow_vis)
    heatmap = get_heatmap(info, args)
    vis_heatmap(f"{path}heatmap.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())
    
def read_video_decord(video_path: str):
  vr = VideoReader(video_path)
  frames = vr.get_batch(list(range(len(vr))))
  return frames
  
@torch.no_grad()
def flow_video(model, args, device=torch.device('cuda')):

    video_path = './assets/videos/rubiks_cube.mp4'
    output_path = './assets/videos/rubiks_cube_flow_lines.mp4'
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    ret, frame1 = cap.read()
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    while ret:
        # Capture frame-by-frame
        ret, frame2 = cap.read()
        if not ret:
            break
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        image1 = torch.tensor(frame1, dtype=torch.float32).permute(2, 0, 1)
        image2 = torch.tensor(frame2, dtype=torch.float32).permute(2, 0, 1)

        H, W = image1.shape[1:]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        H, W = image1.shape[2:]
        flow, info = calc_flow(args, model, image1, image2)     # flow returned is (2, 1080, 1920)
        # flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)

        flow = flow[0].permute(2, 1, 0)             # flow is now (1920, 1080, 2)
        flow_mag = torch.norm(flow, p=1, dim=-1)    # flow_mag is (1920, 1080)
        thresholded_flow = flow[flow_mag > 20]      # thresholded_flow is flattened to (num_points, 2)
        
        H, W = flow.shape[:2]       # H = 1920, W = 1080
        
        uu, vv = torch.meshgrid(torch.arange(H), torch.arange(W))
        indices = torch.stack((uu, vv), dim=-1)[flow_mag > 20]      # indices are taken from the meshgrid, same size as thresholded glow
        correspondences = (indices + thresholded_flow).long().cpu().numpy()     
        
        # new_image1 = frame1[correspondences[..., 0], correspondences[..., 1]]

        lines_img = copy.deepcopy(frame1)

        for i in tqdm(range(0, correspondences.shape[0], 1000)):
            cv2.line(lines_img, indices[i].numpy(), correspondences[i], color = [0, 255, 0], thickness = 2)

        # new_image = copy.deepcopy(frame1)
        # new_image[flow_mag < 20] = 0.

        out.write(lines_img)
        
        # cv2.imwrite(f"{path}flow.jpg", flow_vis)
        # heatmap = get_heatmap(info, args)
        # vis_heatmap(f"{path}heatmap.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())

        frame1 = copy.deepcopy(frame2)        

    # Release the video capture object
    cap.release()
    out.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    # demo_data('./custom/', args, model, image1, image2)

@torch.no_grad()
def demo_custom(model, args, device=torch.device('cuda')):
    # # image1 = cv2.imread("./custom/image1.jpg")
    # # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    # image1 = pillow_heif.open_heif("./custom/IMG_0003.HEIC")
    # image1 = np.asarray(image1)
    # # image2 = cv2.imread("./custom/image2.jpg")
    # # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    # image2 = pillow_heif.open_heif("./custom/IMG_0004.HEIC")
    # image2 = np.asarray(image2)
    # image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    # image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    
    frames = read_video_decord("assets/videos/rubiks_cube.mp4") #.permute(0, 3, 1, 2)
    frame_idx = 26
    frame1 = frames[frame_idx]
    frame2 = frames[frame_idx + 1]
    image1 = frame1.permute(2, 0, 1)[None].to(device)
    image2 = frame2.permute(2, 0, 1)[None].to(device)
    
    flow, _ = calc_flow(args, model, image1, image2)
    flow = flow[0].permute(1, 2, 0)

    flow_mag = torch.norm(flow, p=1, dim=-1)
    thresholded_flow = flow[flow_mag > 20]
    
    H, W = flow.shape[:2]
    
    uu, vv = torch.meshgrid(torch.arange(H), torch.arange(W))
    indices = torch.stack((uu, vv), dim=-1)[flow_mag > 20]
    correspondences = (indices + thresholded_flow).long()
    
    new_image1 = frame1[correspondences[..., 0], correspondences[..., 1]].numpy()

    lines_img = frame1.numpy().astype(np.uint8)

    for i in tqdm(range(0, correspondences.shape[0], 2000)):
        cv2.line(lines_img, indices[i].numpy()[::-1], correspondences[i].numpy()[::-1], color = [0, 255, 0], thickness = 1)
    
    cv2.imwrite("custom/predicted_next_correspondence.jpg", new_image1)
    cv2.imwrite("custom/lines_img.jpg", lines_img)
    
    # cv2.imwrite("custom/prev_frame.jpg", frame1.numpy())
    # cv2.imwrite("custom/next_frame.jpg", frame2.numpy())
    

def track_rigid_body(model, args, video_path, device):
    vr = VideoReader(video_path)
    
    for i in range(len(vr) - 1):
        image1 = vr[i].permute(2, 0, 1)[None].to(device)
        image2 = vr[i + 1].permute(2, 0, 1)[None].to(device)

        # 1. Get optical flow (H, W, 2) (Optional: could threshold here too)
        flow, _ = calc_flow(args, model, image1, image2)  # (1, 2, H, W)    

        H, W = flow.shape[2:]
        
        # 2. For each pixel (u, v), calculate the predicted point correspondence from the optical flow
        uu, vv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        indices = torch.stack((uu, vv), dim=0)
        correspondences = (indices + flow[0]).long()  # Use this to index into image2

        print("")

        # 3. From point correspondences, calculate camera matrices
        # 3.1. Calculate F with either 8pt or 7pt alg (RANSAC)
        # 3.2. Calculate e' from F
        # 3.3. Recover P = [I | 0} and P' = [[e']_\times F | e'] 
        # TODO: figure out how to get all subsequent cameras in the first's coordinate frame
        # 4. From sequence of camera matrices, use triangulation to get reconstruction of the scene
        # 5. There are 2 ways to triangulate:
        #     (a) Use a sliding window of 3 to take triplets of camera matrices and triangulate a pointmap for each
        #     (b) over-constrain the system by triangulating all cameras at once
        # 6. After triangulating, autocalibrate to a matric reconstruction by finding the homography, since we know intrinsics
        # 7. Convert camera delta poses to object delta poses to visualize motion
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')
    args = parse_args(parser)

    model = RAFT.from_pretrained(args.url, args=args)
        
    if args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    # demo_custom(model, args, device=device)
    flow_video(model, args, device=device)

if __name__ == '__main__':
    main()