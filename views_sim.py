import torch
from decord import VideoReader
from sea_raft.raft import RAFT
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

from flow_utils import predict_flow_images, correspondences_from_flow_mask, create_flow_mask, visualized_3d_2frames
from views import plot_epipolar_lines

torch.autograd.set_detect_anomaly(True)

COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (0, 165, 255),  # Orange
    (255, 0, 255),  # Magenta (Pink)
    (0, 255, 255),  # Yellow
    (255, 255, 0),  # Cyan
    (128, 0, 128)   # Purple
]

class Camera(nn.Module):
    def __init__(self, r, t, requires_grad=True):

        super().__init__()

        self.r = nn.Parameter(torch.tensor(r, dtype = torch.float32), requires_grad=requires_grad)        
        self.t = nn.Parameter(torch.tensor(t, dtype = torch.float32), requires_grad=requires_grad)

        # self.register_buffer("r", torch.tensor(r, dtype = torch.float32))
        # self.register_buffer("t", torch.tensor(t, dtype = torch.float32))

    def convert_to_rot_matrix(self):
        """
        Converts a Rodrigues vector to a rotation matrix in a differentiable manner.

        Args:
            rvec (torch.Tensor): Tensor of shape (3,) or (N, 3) containing Rodrigues parameters.

        Returns:
            torch.Tensor: Rotation matrix of shape (3, 3) or (N, 3, 3).
        """
        # r_cv2 = cv2.Rodrigues(self.r.cpu().detach().numpy())
        
        batch = self.r.ndim == 2
        if not batch:
            r = self.r.unsqueeze(0)  # Add batch dimension for uniformity

        theta = torch.norm(r, dim=1, keepdim=True)  # Magnitude of the Rodrigues vector
        r_hat = r / (theta + 1e-8)  # Unit axis of rotation
        x, y, z = r_hat[:, 0], r_hat[:, 1], r_hat[:, 2]

        # Skew-symmetric matrix
        K = torch.zeros(r.shape[0], 3, 3, device=r.device, dtype=r.dtype)
        K[:, 0, 1], K[:, 0, 2] = -z, y
        K[:, 1, 0], K[:, 1, 2] = z, -x
        K[:, 2, 0], K[:, 2, 1] = -y, x

        # Rodrigues formula
        I = torch.eye(3, device=r.device, dtype=r.dtype).unsqueeze(0)
        R = I + torch.sin(theta).unsqueeze(2) * K + (1 - torch.cos(theta).unsqueeze(2)) * torch.bmm(K, K)

        return R.squeeze(0)
    
    def homogenize(self, v):
        ones = torch.ones((*v.shape[:-1], 1), device=v.device)
        return torch.cat((v, ones), dim=-1)
    
    def forward(self, K, X):

        R = self.convert_to_rot_matrix()

        x = torch.matmul(K,
                            torch.matmul(R, X.t()) + self.t)
        x = x / (x[-1] + 1e-8)
        x = x.t()
        x = x[:, :-1]

        return x
    
    @torch.no_grad()
    def camera_matrix(self):
        self.convert_to_rot_matrix()
        return torch.cat((self.R, self.t[:, None]), axis=-1)
    
class GlobalOptimization(nn.Module):
    def __init__(self, K, x0, x1, x2, X, r1, t1, r2, t2, r0 = None, t0 = None, cam1_requires_grad=True):

        super().__init__()

        self.register_buffer("x0", torch.tensor(x0, dtype = torch.float32))
        self.register_buffer("x1", torch.tensor(x1, dtype = torch.float32))
        self.register_buffer("x2", torch.tensor(x2, dtype = torch.float32))
        self.register_buffer("K", torch.tensor(K, dtype = torch.float32))

        self.X = nn.Parameter(torch.tensor(X, dtype = torch.float32))
        # self.register_buffer("X", torch.tensor(X, dtype = torch.float32))

        if r0 is None:
            r0 = np.zeros(3)

        if t0 is None:
            t0 = np.zeros((3, 1))

        self.cam0 = Camera(r0, t0, requires_grad=False)
        self.cam1 = Camera(r1, t1, requires_grad=cam1_requires_grad)
        self.cam2 = Camera(r2, t2)
    
    def forward(self):
        # x0_pred = torch.matmul(self.K, self.X.t())
        # x0_pred = x0_pred / x0_pred[-1]
        # x0_pred = x0_pred.t()
        # x0_pred = x0_pred[:, :-1]

        x0_pred = self.cam0(self.K, self.X)
        x1_pred = self.cam1(self.K, self.X)
        x2_pred = self.cam2(self.K, self.X)

        loss1 = torch.norm(self.x0 - x0_pred, p=2, dim=-1) ** 2
        loss2 = torch.norm(self.x1 - x1_pred, p=2, dim=-1) ** 2
        loss3 = torch.norm(self.x2 - x2_pred, p=2, dim=-1) ** 2

        loss = (loss1 + loss2 + loss3).mean()

        return loss
    
    @torch.no_grad()
    def get_params(self):
        X = self.X.cpu().detach().numpy()
        R1 = self.cam1.convert_to_rot_matrix().cpu().detach().numpy()
        R2 = self.cam2.convert_to_rot_matrix().cpu().detach().numpy()
        t1 = self.cam1.t.cpu().detach().numpy()
        t2 = self.cam2.t.cpu().detach().numpy()
        return X, R1, R2, t1, t2
    
    
def homogenize_2d(pts):
    
    assert pts.shape[-1] == 3 or pts.shape[-1] == 2

    if pts.shape[-1] == 2:
        return np.append(pts, np.ones((*pts.shape[:-1], 1)), axis = -1)
    else:
        return pts
    
    
def epipolar_lines(F, pts1, pts2):

    pts1 = homogenize_2d(pts1)
    pts2 = homogenize_2d(pts2)
    
    l1 = ((F.T) @ (pts2.T)).T
    l2 = (F @ (pts1.T)).T

    return l1, l2


def find_line_intersect_with_borders(line, img):

    height, width = img.shape[:2]
    a, b, c = line[:, 0], line[:, 1], line[:, 2]

    x0 = np.zeros_like(a)
    y0 = -c/b
    x1 = np.ones_like(a) * width-1
    y1 = (-a * (width-1) -c) / b

    line_pts = np.column_stack([x0, y0, x1, y1])
    
    return line_pts


def plot_epipolar_lines(F, pts1, pts2, img1, img2):

    el1, el2 = epipolar_lines(F, pts1, pts2)

    el1_pts = find_line_intersect_with_borders(el1, img1)
    el2_pts = find_line_intersect_with_borders(el2, img2)

    img1_pts = img1.copy()
    for i in range(pts1.shape[0]):
        cv2.circle(img1_pts, tuple(pts1[i, :2].astype(np.int64)), radius=8, color=COLORS[i], thickness=-1)

    # cv2.imshow('Viewpoint 1 Points', img1_pts)
    cv2.imwrite('img1_pts.jpg', img1_pts)
    
    # plt.imshow(img1_pts)
    # plt.show()
    # cv2.waitKey(0)

    img2_lines = img2.copy()
    for i in range(el2_pts.shape[0]):
        line = el2_pts[i].astype(np.int64)
        cv2.line(img2_lines, (line[0], line[1]), (line[2], line[3]), color=COLORS[i], thickness=4)     # line is arranged as (x0, y0, x1, y1)

    cv2.imwrite('img2_lines.jpg', img2_lines)
    # cv2.imshow('Viewpoint 2 Lines', img2_lines)
    # cv2.waitKey(0)

    # plt.imshow(img2_lines)
    # plt.show()

    img2_pts = img2.copy()
    for i in range(pts2.shape[0]):
        cv2.circle(img2_pts, tuple(pts2[i, :2].astype(np.int64)), radius=8, color=COLORS[i], thickness=-1)

    cv2.imwrite('img2_pts.jpg', img2_pts)
    # cv2.imshow('Viewpoint 2 Points', img2_pts)
    # cv2.waitKey(0)

    # plt.imshow(img2_pts)
    # plt.show()

    img1_lines = img1.copy()
    for i in range(el1_pts.shape[0]):
        line = el1_pts[i].astype(np.int64)
        cv2.line(img1_lines, (line[0], line[1]), (line[2], line[3]), color=COLORS[i], thickness=4)     # line is arranged as (x0, y0, x1, y1)

    # cv2.imshow('Viewpoint 1 Lines', img1_lines)
    # cv2.waitKey(0)

    cv2.imwrite('img1_lines.jpg', img1_lines)
    # plt.imshow(img1_lines)
    # plt.show()

    # cv2.destroyAllWindows()

    return img1_pts, img2_pts, img1_lines, img2_lines
        

def reconstruct_3_frames(K, R0, t0, video, i, flow_model, args, R1 = None, t1 = None):
    prev_frame = video[i, None]
    curr_frame = video[i + args.skip*1, None]
    next_frame = video[i + args.skip*2, None]

    # i is the index of the 0th frame

    pts3d = np.load(args.video_path[:-4] + "_pcd.npy")
    pts3d_seg = np.load(args.video_path[:-4] + "_seg.npy")

    seg0, seg1, seg2 = pts3d_seg[i] == 2, pts3d_seg[i + args.skip*1] == 2, pts3d_seg[i + args.skip*2] == 2

    is_beginning = False

    if R1 is None and t1 is None:
        is_beginning = True

    # Flows
    back_flow = predict_flow_images(curr_frame, prev_frame, flow_model, args)
    forward_flow = predict_flow_images(curr_frame, next_frame, flow_model, args)

    # Masking
    back_flow_mask = create_flow_mask(back_flow, args)
    forward_flow_mask = create_flow_mask(forward_flow, args)

    # Correspondences
    correspondence_mask = (back_flow_mask * forward_flow_mask).bool()[0]
    rgb = curr_frame[:, correspondence_mask].cpu().detach().numpy()[0] / 255.

    x1, x2 = correspondences_from_flow_mask(forward_flow[0], correspondence_mask, args)
    x1, x0 = correspondences_from_flow_mask(back_flow[0], correspondence_mask, args)

    x0 = x0.cpu().detach().numpy().astype(np.float32)
    x1 = x1.cpu().detach().numpy().astype(np.float32)
    x2 = x2.cpu().detach().numpy().astype(np.float32)

    x0_seg = seg0[x0[:, 1].astype(int), x0[:, 0].astype(int)]
    x1_seg = seg1[x1[:, 1].astype(int), x1[:, 0].astype(int)]
    x2_seg = seg2[x2[:, 1].astype(int), x2[:, 0].astype(int)]
    # Seg mask works

    seg_mask = x0_seg * x1_seg * x2_seg
    x0, x1, x2 = x0[seg_mask], x1[seg_mask], x2[seg_mask]
    x0 = x0[:, ::-1]
    x1 = x1[:, ::-1]
    x2 = x2[:, ::-1]

    img0 = prev_frame[0].cpu().detach().numpy()
    img1 = curr_frame[0].cpu().detach().numpy()
    img2 = next_frame[0].cpu().detach().numpy()
    lines_img = np.vstack((img0, img1, img2))

    x1_prime = x1.copy()
    x1_prime[:, 1] = x1[:, 1] + img1.shape[0]
    x2_prime = x2.copy()
    x2_prime[:, 1] = x2[:, 1] + 2*img1.shape[0] 

    # Sample 10 random points:
    pt_indices = np.random.choice(x0.shape[0], 10, replace=False)
    x = x0[pt_indices]
    x1_prime = x1_prime[pt_indices]
    x2_prime = x2_prime[pt_indices]

    for k in range(x.shape[0]):
        cv2.line(lines_img, x0[k, :].astype(np.int64), x1_prime[k, :].astype(np.int64), color = [0, 255, 0], thickness = 3)
        cv2.line(lines_img, x1_prime[k, :].astype(np.int64), x2_prime[k, :].astype(np.int64), color = [0, 255, 0], thickness = 3)

    cv2.imwrite("lines_img.jpg", lines_img)

    # Look inside the get_pointcloud() function,
    # x --> width
    # y --> height (this is probably the opposite of our correspondences)
    X3D = pts3d[x0[:, 1].astype(int), x0[:, 0].astype(int)]
    rgb = rgb[seg_mask]

    #obj_id = 2

    # First camera
    P0 = K @ np.hstack((R0, t0))
    
    # Get 2nd camera:
    ret, rvecs, tvecs = cv2.solvePnP(X3D, x1[:, ::-1], K, None)
    R1 = cv2.Rodrigues(rvecs)[0]
    t1 = tvecs

    # Get 3rd camera:
    ret, rvecs, tvecs = cv2.solvePnP(X3D, x2[:, ::-1], K, None)
    R2 = cv2.Rodrigues(rvecs)[0]
    t2 = tvecs

    # Recover cameras:
    # P0 = K @ P0
    P1 = K @ np.hstack((R1, t1)) 

    # Triangulate:
    # X3D = cv2.triangulatePoints(P0, P1, x0.T, x1.T)
    # X3D = X3D[:3, :] / X3D[3, :]
    # X3D = X3D.T

    # Visualize:
    geometries = visualized_3d_2frames(X3D, R1, t1[:, 0], size = 0.05)
    o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

    # Visualize:
    # geometries = visualized_3d_3frames(X3D, R1, t1[:, 0], R2, t2)
    # # o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

    r0 = cv2.Rodrigues(R0)[0][:, 0]
    r1 = cv2.Rodrigues(R1)[0][:, 0]
    r2 = rvecs[:, 0]
    global_opt = GlobalOptimization(K, x0, x1, x2, X3D, r1, t1, r2, t2, r0, t0, cam1_requires_grad=is_beginning)
    global_opt.to(args.device)
    optimizer = torch.optim.AdamW(global_opt.parameters(), lr = args.lr)

    for _ in tqdm(range(args.iterations)):
        optimizer.zero_grad()
        loss = global_opt()
        loss.backward()
        optimizer.step()

    X, R1, R2, t1, t2 = global_opt.get_params()

    return X, R1, R2, t1, t2, rgb