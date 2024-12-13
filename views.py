import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from tqdm import tqdm

from flow_utils import predict_flow_images, correspondences_from_flow_mask, create_flow_mask, visualized_3d_2frames


torch.autograd.set_detect_anomaly(True)


class Camera(nn.Module):
    def __init__(self, r, t, requires_grad=True):
        super().__init__()

        self.r = nn.Parameter(torch.tensor(r, dtype = torch.float32), requires_grad=requires_grad)        
        self.t = nn.Parameter(torch.tensor(t, dtype = torch.float32), requires_grad=requires_grad)

    def convert_to_rot_matrix(self):
        """
        Converts a Rodrigues vector to a rotation matrix in a differentiable manner.

        Args:
            rvec (torch.Tensor): Tensor of shape (3,) or (N, 3) containing Rodrigues parameters.

        Returns:
            torch.Tensor: Rotation matrix of shape (3, 3) or (N, 3, 3).
        """
        
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

        if r0 is None:
            r0 = np.zeros(3)

        if t0 is None:
            t0 = np.zeros((3, 1))

        self.cam0 = Camera(r0, t0, requires_grad=False)
        self.cam1 = Camera(r1, t1, requires_grad=cam1_requires_grad)
        self.cam2 = Camera(r2, t2)
    
    def forward(self):
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


def reconstruct_3_frames(K, R0, t0, video, i, flow_model, args, R1 = None, t1 = None):
    prev_frame = video[i, None]
    curr_frame = video[i + args.skip*1, None]
    next_frame = video[i + args.skip*2, None]

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

    P0 = np.hstack((R0, t0))

    # Recover pose of first camera
    if is_beginning:
        # Get Essential Matrix
        E, _ = cv2.findEssentialMat(x0, x1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Recover second camera pose
        _, R1, t1, _ = cv2.recoverPose(E, x0, x1, K)

    # Recover cameras:
    P0 = K @ P0
    P1 = K @ np.hstack((R1, t1)) 

    # Triangulate:
    X3D = cv2.triangulatePoints(P0, P1, x0.T, x1.T)
    X3D = X3D[:3, :] / X3D[3, :]
    X3D = X3D.T

    ## Uncomment if you want to see the initial 3D reconstruction and camera frames
    # if is_beginning:
    #     geometries = visualized_3d_2frames(X3D, R1, t1[:, 0], rgb_ = rgb)
    #     o3d.visualization.draw_geometries(geometries, window_name="Initial Reconstruction")

    # Get 3rd camera:
    _, rvecs, tvecs = cv2.solvePnP(X3D, x2, K, None)
    R2 = cv2.Rodrigues(rvecs)[0]
    t2 = tvecs

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
