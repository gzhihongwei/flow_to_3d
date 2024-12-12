import torch
import torch.nn.functional as F
import numpy as np
import cv2
import open3d as o3d
import decord
from tqdm import tqdm
decord.bridge.set_bridge("torch")

from decord import VideoReader

from sea_raft.raft import RAFT
from sea_raft.utils.flow_viz import flow_to_image

def read_video(args):
  vr = VideoReader(args.video_path)
  frames = vr.get_batch(list(range(len(vr))))
  frames = frames.to(args.device)
  return frames

def load_flow_model(args):
  model = RAFT.from_pretrained(args.url, args=args)
  model = model.to(args.device)
  model.eval()

  return model

@torch.no_grad()
def forward_flow(args, model, image1, image2):
  """
  Forward pass of sea raft flow model
  """
  output = model(image1, image2, iters=args.iters, test_mode=True)
  flow_final = output['flow'][-1]
  info_final = output['info'][-1]
  return flow_final, info_final

def calc_flow(args, model, image1, image2):
  """
  Postprocessing after sea raft flow model prediction
  """

  img1 = F.interpolate(image1.to(torch.float32), scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
  img2 = F.interpolate(image2.to(torch.float32), scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
  H, W = img1.shape[2:]
  flow, info = forward_flow(args, model, img1, img2)
  flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
  info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
  return flow_down, info_down

def predict_flow(video, model, args):
  """
  video => torch tensor of shape (frames, H, W, 3)
  """

  vid_input = torch.permute(video, (0, 3, 1, 2))    # Now vid_input is (frames, 3, H, W)   (0, 2, 3, 1)

  frame1 = vid_input[:-args.skip]
  frame2 = vid_input[args.skip:]

  flow, _ = calc_flow(args, model, frame1, frame2)    # Flow is predicted in (frames, 2, H, W)
  flow = torch.permute(flow, (0, 2, 3, 1))            # Now flow is (frames, H, W, 2)

  return flow

def predict_flow_images(image1, image2, model, args):
  """
  video => torch tensor of shape (frames, H, W, 3)
  """

  image1 = torch.permute(image1, (0, 3, 1, 2))
  image2 = torch.permute(image2, (0, 3, 1, 2))
  flow, _ = calc_flow(args, model, image1, image2)    # Flow is predicted in (frames, 2, H, W)
  flow = torch.permute(flow, (0, 2, 3, 1))            # Now flow is (frames, H, W, 2)

  return flow

def create_flow_heatmap(video, model, args):
  """
  flow => torch tensor of shape (frames, H, W, 2)
  """

  cap = cv2.VideoCapture(args.video_path)
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  
  output_path = args.video_path[:-4] + "_heatmap.mp4"      # Everything before .mp4 + _heatmap.mp4

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

  # ------------------------------- #
  num_frames = video.shape[0]
  num_batches = num_frames // args.batch + (0 if (num_frames % args.batch == 0) else 1)

  print("Creating Heatmap:")
  for i in tqdm(range(num_batches)):

    flow = predict_flow(video[(i * args.batch) : ((i+1) * args.batch)], model, args)
    
    for j in range(flow.shape[0]):
      
      flow_vis = flow_to_image(flow[j].cpu().detach().numpy(), convert_to_bgr=True)
      writer.write(flow_vis)

  cap.release()
  writer.release()

def create_flow_mask_heatmap(video, model, args):
  """
  flow => torch tensor of shape (frames, H, W, 2)
  """

  cap = cv2.VideoCapture(args.video_path)
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  
  output_path = args.video_path[:-4] + "_masked.mp4"      # Everything before .mp4 + _heatmap.mp4

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

  # ------------------------------- #
  num_frames = video.shape[0]
  num_batches = num_frames // args.batch + (0 if (num_frames % args.batch == 0) else 1)

  print("Creating Heatmap:")
  for i in tqdm(range(num_batches)):

    flow = predict_flow(video[(i * args.batch) : ((i+1) * args.batch)], model, args)
    flow_mask = create_flow_mask(flow, args)    # flow_mask is (frames, H, W)
    masked_flow = flow * flow_mask.unsqueeze(-1).float()
    
    for j in range(flow.shape[0]):
      
      flow_vis = flow_to_image(masked_flow[j].cpu().detach().numpy(), convert_to_bgr=True)
      writer.write(flow_vis)

  cap.release()
  writer.release()

def create_flow_correspondence_video(video, model, args):
  """
  flow => torch tensor of shape (frames, H, W, 2)
  """

  cap = cv2.VideoCapture(args.video_path)
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 2
  fps = cap.get(cv2.CAP_PROP_FPS)
  
  output_path = args.video_path[:-4] + "_correspondences.mp4"      # Everything before .mp4 + _heatmap.mp4

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

  # ------------------------------- #
  num_frames = video.shape[0]
  num_batches = num_frames // args.batch + (0 if (num_frames % args.batch == 0) else 1)

  print("Creating Heatmap:")
  for i in tqdm(range(num_batches)):

    flow = predict_flow(video[(i * args.batch) : ((i+1) * args.batch)], model, args)
    
    for j in range(flow.shape[0]):

      img1 = video[(i * args.batch) + j].cpu().detach().numpy()
      img2 = video[(i * args.batch) + j + 1].cpu().detach().numpy()
      lines_img = np.vstack((img1, img2))

      x, x_prime = correspondences_from_flow(flow[j], args)

      if x.size(0) >= 10:

        x = x.cpu().detach().numpy()
        x_prime = x_prime.cpu().detach().numpy()

        x_prime[:, 1] = x_prime[:, 1] + img1.shape[0]     # Shift downwards because images are stacked

        # Sample 10 random points:
        pt_indices = np.random.choice(x.shape[0], 10, replace=False)
        x = x[pt_indices]
        x_prime = x_prime[pt_indices]

        for k in range(x.shape[0]):
          cv2.line(lines_img, x[k, :].astype(np.int64), x_prime[k, :].astype(np.int64), color = [0, 255, 0], thickness = 3)

        
        lines_img = cv2.cvtColor(lines_img, cv2.COLOR_BGR2RGB)
        writer.write(lines_img)

  cap.release()
  writer.release()

def create_flow_mask(flow, args):
  """
  flow => torch tensor of shape (frames, H, W, 2)
  """

  flow_mag = torch.norm(flow, p=2, dim=-1) # flow_mag is (batch, H, W)
  flow_mask = flow_mag > args.thresh

  return flow_mask

def correspondences_from_flow(flow, args):
  """
  flow => torch tensor of shape (H, W, 2)  [Remember to index into flow to get this]
  """
  
  H, W = flow.shape[:2]

  flow_mask = create_flow_mask(flow, args)
  
  vv, uu = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

  x = torch.stack((uu, vv), dim=-1).to(args.device)
  x = x[flow_mask]
  x = x.float() 
  x_prime = x + flow[flow_mask]

  # REMEMBER: x & x_prime are in the coordinates (W->1920, H->1080)

  return x, x_prime 

def correspondences_from_flow_mask(flow, flow_mask, args):
  """
  flow => torch tensor of shape (H, W, 2)  [Remember to index into flow to get this]
  """
  
  H, W = flow.shape[:2]
  
  vv, uu = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

  x = torch.stack((uu, vv), dim=-1).to(args.device)
  x = x[flow_mask]
  x = x.float() 
  x_prime = x + flow[flow_mask]

  # REMEMBER: x & x_prime are in the coordinates (W->1920, H->1080)

  return x, x_prime 

def visualized_3d_2frames(X3D, R, t, size=2., rgb_ = None):

  T1 = np.eye(4)
  T1[:3, :3] = R
  T1[:3, 3] = t

  geometries = []
  frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
  frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
  frame2.transform(T1.copy())

  geometries.append(frame1)
  geometries.append(frame2)

  pcd = np.zeros((X3D.shape[0], 3))
  pcd[:, :] = X3D[:, :]
  pts_vis = o3d.geometry.PointCloud()
  pts_vis.points = o3d.utility.Vector3dVector(pcd)

  if rgb_ is not None:
    rgb = np.zeros((X3D.shape[0], 3))
    rgb[:, :] = rgb_
    pts_vis.colors = o3d.utility.Vector3dVector(rgb)
    
  geometries.append(pts_vis)

  return geometries

def visualized_3d_3frames(X3D, camera_poses, rgb_):
  geometries = []
  frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2., origin=[0, 0, 0])
  geometries.append(frame)

  for R, t in camera_poses:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3,] = t.squeeze()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2., origin=[0, 0, 0])
    frame.transform(T.copy())
    geometries.append(frame)

  pcd = np.zeros((X3D.shape[0], 3))
  pcd[:, :] = X3D[:, :]

  rgb = np.zeros((X3D.shape[0], 3))
  rgb[:, :] = rgb_
  pts_vis = o3d.geometry.PointCloud()
  pts_vis.points = o3d.utility.Vector3dVector(pcd)
  pts_vis.colors = o3d.utility.Vector3dVector(rgb)
  geometries.append(pts_vis)

  return geometries