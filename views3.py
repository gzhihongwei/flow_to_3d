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

from flow_utils import predict_flow_images, correspondences_from_flow_mask, create_flow_mask

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
    def __init__(self, r, t):

        super().__init__()

        self.r = nn.Parameter(torch.tensor(r, dtype = torch.float32))        
        self.t = nn.Parameter(torch.tensor(t, dtype = torch.float32))

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
        r_cv2 = cv2.Rodrigues(self.r.cpu().detach().numpy())
        
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
                            torch.matmul(R, X.t()) + self.t.unsqueeze(-1))
        x = x / x[-1]
        x = x.t()
        x = x[:, :-1]

        return x
    
    @torch.no_grad()
    def camera_matrix(self):
        self.convert_to_rot_matrix()
        return torch.cat((self.R, self.t[:, None]), axis=-1)
    
class GlobalOptimization(nn.Module):
    def __init__(self, K, x0, x1, x2, X, r1, t1, r2, t2):

        super().__init__()

        self.register_buffer("x0", torch.tensor(x0, dtype = torch.float32))
        self.register_buffer("x1", torch.tensor(x1, dtype = torch.float32))
        self.register_buffer("x2", torch.tensor(x2, dtype = torch.float32))
        self.register_buffer("K", torch.tensor(K, dtype = torch.float32))

        self.X = nn.Parameter(torch.tensor(X, dtype = torch.float32))
        # self.register_buffer("X", torch.tensor(X, dtype = torch.float32))

        self.cam1 = Camera(r1, t1)
        self.cam2 = Camera(r2, t2)
    
    def forward(self):

        x0_pred = torch.matmul(self.K, self.X.t())
        x0_pred = x0_pred / x0_pred[-1]
        x0_pred = x0_pred.t()
        x0_pred = x0_pred[:, :-1]

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
        
    
    @torch.no_grad()
    def triangulate(self, rgb = None):
        self.set_K()

        P1 = torch.zeros((1, 3, 4), dtype=torch.float32, device = "cuda")
        P1[:, :3, :3] = self.K

        P2 = torch.tensor(self.cameras.camera_matrix(), dtype=torch.float32, device = "cuda").unsqueeze(0)
        P2 = torch.matmul(self.K.cuda(), P2)

        x = self.x.cuda()
        x_prime = self.x_prime.cuda()

        x1_skew = skew_symmetric_matrix(homogenize(x))
        x2_skew = skew_symmetric_matrix(homogenize(x_prime))

        constraint1 = torch.matmul(x1_skew[..., :-1, :], P1)
        constraint2 = torch.matmul(x2_skew[..., :-1, :], P2)

        constraint_matrix = torch.concat([constraint1, constraint2], dim = -2)
        U, S, Vh = torch.linalg.svd(constraint_matrix)
        X = Vh[..., -1, :]

        pts3d = (X / X[:, -1, None])[:, :-1].cpu()

        pcd = np.zeros((pts3d.size(0), 3))
        pcd[:, :] = pts3d.cpu().detach().numpy()

        if rgb is not None:
            colors = np.zeros((X.size(0), 3))
            colors[:, :] = (rgb / 255.).cpu().detach().numpy()

        geometries = []
        pts_vis = o3d.geometry.PointCloud()
        pts_vis.points = o3d.utility.Vector3dVector(pcd)
        if rgb is not None:
            pts_vis.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pts_vis)

        o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

        return pts3d

    @torch.no_grad()
    def plot(self, rgb = None):

        X = self.depth.abs() * torch.matmul(torch.linalg.inv(self.K), self.homogenize(self.x).t())

        pcd = np.zeros((X.size(0), 3))
        pcd[:, :] = X.cpu().detach().numpy()

        if rgb is not None:
            colors = np.zeros((X.size(0), 3))
            colors[:, :] = (rgb / 255.).cpu().detach().numpy()

        geometries = []
        pts_vis = o3d.geometry.PointCloud()
        pts_vis.points = o3d.utility.Vector3dVector(pcd)
        if rgb is not None:
            pts_vis.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pts_vis)

        o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

        return geometries

def homogenize_2d(pts):
    
    assert pts.shape[-1] == 3 or pts.shape[-1] == 2

    if pts.shape[-1] == 2:
        return np.append(pts, np.ones((*pts.shape[:-1], 1)), axis = -1)
    else:
        return pts

def skew_symmetric_matrix(v, mode = "torch"):

    if mode == 'torch':
        matrix = torch.zeros((*v.shape[:-1], 3, 3), device=v.device)
    elif mode == "numpy":
        matrix = np.zeros((*v.shape[:-1], 3, 3))
    
    matrix[..., 0, 1] = -v[..., 2]
    matrix[..., 0, 2] = v[..., 1]
    matrix[..., 1, 2] = -v[..., 0]

    matrix[..., 1, 0] = v[..., 2]
    matrix[..., 2, 0] = -v[..., 1]
    matrix[..., 2, 1] = v[..., 0]

    return matrix

def get_correspondences(flow, image):

    H = flow.size(1)
    W = flow.size(2)
    image = image.permute(0, 2, 3, 1)

    flow_mag = torch.norm(flow, p=1, dim=-1)
    flow_mask = (flow_mag > 20)
    # self.flow_mask = (flow_mag > 20).float()
    
    vv, uu = torch.meshgrid(torch.arange(H, device=flow.device), torch.arange(W, device=flow.device), indexing="ij")

    # Principal point is (0,0):
    vv = vv - H/2
    uu = uu - W/2

    indices = torch.stack((uu, vv), dim=0).unsqueeze(0)
    indices = indices.permute(0, 2, 3, 1)

    # Note; Indices are switched and added to flow, because the flow predictions are switched => (x, y) corresponds to (W, H)
    # homogenized_inds = homogenize(indices)

    correspondences = indices + flow
    # homogenized_corr = homogenize(correspondences)

    indices = indices[flow_mask]
    correspondences = correspondences[flow_mask]
    rgb = image[flow_mask]
    
def homogenize(v):
    ones = np.ones((*v.shape[:-1], 1))
    return np.concatenate((v, ones), axis=-1)

def F_to_E(F):

    U, S, Vh = np.linalg.svd(F)

    S_ones = np.eye(3)
    S_ones[-1, -1] = 0

    E = U @ S_ones @ Vh

    return E


def reconstruct_3_frames(K, video, i, flow_model, args):
    prev_frame = video[i, None]
    curr_frame = video[i + 2, None]
    next_frame = video[i + 4, None]

    # Flows
    back_flow = predict_flow_images(curr_frame, prev_frame, flow_model, args)
    forward_flow = predict_flow_images(curr_frame, next_frame, flow_model, args)

    # Masking
    back_flow_mask = create_flow_mask(back_flow, args)
    forward_flow_mask = create_flow_mask(forward_flow, args)

    # Correspondences
    correspondence_mask = (back_flow_mask * forward_flow_mask).bool()[0]

    x1, x0 = correspondences_from_flow_mask(back_flow[0], correspondence_mask, args)
    x1, x2 = correspondences_from_flow_mask(forward_flow[0], correspondence_mask, args)

    x0 = x0.cpu().detach().numpy().astype(np.float32)
    x1 = x1.cpu().detach().numpy().astype(np.float32)
    x2 = x2.cpu().detach().numpy().astype(np.float32)

    # Get Essential Matrix
    F, _ = cv2.findFundamentalMat(x0, x1, method=cv2.FM_8POINT)
    E = K.T @ F @ K
    E = F_to_E(E)

    # Recover pose of first camera
    _, R1, t1, mask = cv2.recoverPose(E, x0, x1, K)

    # Recover cameras:
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera
    P1 = K @ np.hstack((R1, t1)) 

    # Triangulate:
    X3D = cv2.triangulatePoints(P0, P1, x0.T, x1.T)
    X3D = X3D[:3, :] / X3D[3, :]
    X3D = X3D.T

    # Visualize:
    # geometries = visualized_3d_2frames(X3D, R1, t1[:, 0])
    # # o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

    # Get 3rd camera:
    ret, rvecs, tvecs = cv2.solvePnP(X3D, x2, K, None)
    R2 = cv2.Rodrigues(rvecs)[0]
    t2 = tvecs[:, 0]

    # Visualize:
    # geometries = visualized_3d_3frames(X3D, R1, t1[:, 0], R2, t2)
    # # o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

    r1 = cv2.Rodrigues(R1)[0][:, 0]
    r2 = rvecs[:, 0]
    global_opt = GlobalOptimization(K, x0, x1, x2, X3D, r1, r2, t1[:, 0], t2)
    global_opt.to(args.device)
    optimizer = torch.optim.AdamW(global_opt.parameters(), lr = args.lr)

    # losses = []
    for _ in tqdm(range(args.iterations)):
        optimizer.zero_grad()
        loss = global_opt()
        loss.backward()
        optimizer.step()
        # losses.append(loss.item())

    # print(loss.item())
    # plt.plot(losses[1000:])
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()

    X, R1, R2, t1, t2 = global_opt.get_params()

    return X, R1, R2, t1, t2

    # # Visualize:
    # geometries = visualized_3d_3frames(X, R1, t1, R2, t2)
    # o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")


def invert_similarity_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv