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

class Views(nn.Module):

    def __init__(self, args, flow) -> None:

        super(Views, self).__init__()

        self.args = args
        self.device = args.device
        self.register_buffer('flow', flow)
        
        # self.register_buffer('frames', self.read_video_decord(video)[20:25]) # TODO: Remove this!
        self.register_buffer('H', torch.tensor(self.flow.shape[-3]))
        self.register_buffer('W', torch.tensor(self.flow.shape[-2]))

        self.register_buffer('num_frames', torch.tensor(self.flow.size(0) + 1))
        # self.flow_model = RAFT.from_pretrained(args.url, args=args)
        # self.flow_model.to(self.device)
        # self.flow_model.eval()
        
        # self.rots = nn.Parameter(torch.randn(self.num_frames-1, 2, 3))      # 6D rotation representation, (v1, v2)
        self.rots = nn.Parameter(torch.eye(3)[:-1].unsqueeze(0).expand(self.num_frames - 1, -1, -1))      # 6D rotation representation, (v1, v2)
        # self.trans = nn.Parameter(torch.randn(self.num_frames-1, 3))
        self.register_buffer('trans', torch.tensor([[1., 1., 0.1]]).expand(self.num_frames - 1, -1,))
        self.focal = nn.Parameter(torch.abs(200*torch.randn(1)))

        # Convert focal length to intrinsics:
        self.K = torch.diag_embed(torch.tensor([self.focal, self.focal, 1.]).unsqueeze(0)).to(self.device)
        # self.K = torch.diag(torch.tensor([self.focal, self.focal, 1])).unsqueeze(0)
        # self.K[0, :2, :2] = torch.abs(self.focal)
        # self.K[0, 1, 1] = torch.abs(self.focal)
        self.Kt = torch.transpose(self.K, -1, -2).to(self.device)
        self.Kt_inv = torch.transpose(torch.inverse(self.K), -1, -2).to(self.device)

        # self.K = torch.diag(torch.tensor([torch.abs(self.focal), torch.abs(self.focal), 1])).unsqueeze(0)

        # self.register_buffer('flow', self.get_flow(self.frames))
        # self.flow = self.get_flow(self.frames)

        x, x_prime = self.get_correspondences(self.flow)
        self.register_buffer('x', x)
        self.register_buffer('x_prime', x_prime)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=args.lr)

    def read_video_decord(self, video_path: str):
        vr = VideoReader(video_path)
        frames = vr.get_batch(list(range(len(vr))))
        return frames.permute(0, 3, 1, 2) #.to(self.device)
    
    def get_flow(self, frames):

        frame1 = frames[:-1]
        frame2 = frames[1:]

        print("Calculating Flow")
        flow, info = self.calc_flow(frame1, frame2)
        print("Retrieved flow")

        # flow = flow.reshape(flow.size(0), 2, -1).permute(0, 2, 1)       # batch, 2, h*w
        flow = flow.permute(0, 2, 3, 1)

        return flow     # => torch.Size([4, 1080, 1920, 2])

        # print('Getting Flow:\n')
        # for i in tqdm(range(1, frames.size(0))):

        #     img1 = None

            # Input size to flow; torch.Size([1, 3, 1080, 1920])

    @torch.no_grad()
    def forward_flow(self, image1, image2):
        output = self.flow_model(image1, image2, iters=self.args.iters, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
        return flow_final, info_final

    def calc_flow(self, image1, image2):
        img1 = F.interpolate(image1, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
        H, W = img1.shape[2:]
        flow, info = self.forward_flow(img1, img2)
        flow_down = F.interpolate(flow, scale_factor=0.5 ** self.args.scale, mode='bilinear', align_corners=False) * (0.5 ** self.args.scale)
        info_down = F.interpolate(info, scale_factor=0.5 ** self.args.scale, mode='area')
        return flow_down, info_down
    
    def get_correspondences(self, flow):

        flow_mag = torch.norm(self.flow, p=1, dim=-1)
        self.register_buffer("flow_mask", (flow_mag > 20).float())   
        # self.flow_mask = (flow_mag > 20).float()
        
        vv, uu = torch.meshgrid(torch.arange(self.H, device=self.device), torch.arange(self.W, device=self.device), indexing="ij")

        # Principal point is (0,0):
        vv = vv - self.H/2
        uu = uu - self.W/2

        indices = torch.stack((uu, vv), dim=0).unsqueeze(0)
        indices = indices.permute(0, 2, 3, 1)

        # Note; Indices are switched and added to flow, because the flow predictions are switched => (x, y) corresponds to (W, H)
        homogenized_inds = self.homogenize(indices)

        correspondences = indices + flow
        homogenized_corr = self.homogenize(correspondences)
        
        return homogenized_inds, homogenized_corr
    
    def convert_rot_to_matrix(self, rots):

        v1 = rots[:, 0]
        v2 = rots[:, 1]

        r1 = v1 / v1.norm(dim=-1, keepdim=True)

        r2_proj = v2 - (v2 * r1).sum(dim=-1, keepdim=True) * r1
        r2 = r2_proj / r2_proj.norm(dim=-1, keepdim=True)

        r3 = torch.cross(r1, r2, dim=-1)

        R = torch.stack([r1, r2, r3], dim=-1)
        return R
    
    def skew_symmetric_matrix(self, v):

        matrix = torch.zeros((*v.shape[:-1], 3, 3), device=v.device)
        
        matrix[..., 0, 1] = -v[..., 2]
        matrix[..., 0, 2] = v[..., 1]
        matrix[..., 1, 2] = -v[..., 0]

        matrix[..., 1, 0] = v[..., 2]
        matrix[..., 2, 0] = -v[..., 1]
        matrix[..., 2, 1] = v[..., 0]

        return matrix
    
    def fundamental_matrix(self):
        
        Rt = torch.transpose(self.R, -1, -2)

        KRTt = torch.matmul(self.K, torch.matmul(Rt, self.trans.unsqueeze(-1)))[..., 0]
        KRTt_skew = self.skew_symmetric_matrix(KRTt)

        estimated_F = torch.matmul(self.Kt_inv, torch.matmul(self.R, torch.matmul(self.Kt, KRTt_skew)))

        U, S, Vh = torch.linalg.svd(estimated_F, full_matrices = False)
        S_zero = torch.zeros((U.size(0), 3, 3), device=self.device)
        S_zero[:, 0, 0] = S[:, 0]
        S_zero[:, 1, 1] = S[:, 1]

        F = torch.matmul(U, torch.matmul(S_zero, Vh))

        return F
    
    def homogenize(self, v):
        ones = torch.ones((*v.shape[:-1], 1), device=v.device)
        return torch.cat((v, ones), dim=-1)
    
    def reproj_loss(self):

        estimated_F = self.fundamental_matrix()

        x_flat = self.x.permute(0, 3, 1, 2).reshape(self.x.size(0), 3, -1)
        x_prime_flat = self.x_prime.permute(0, 3, 1, 2).reshape(self.x_prime.size(0), 3, -1)
        flat_flow_mask = self.flow_mask.reshape(self.flow_mask.size(0), -1)

        Fx = torch.matmul(estimated_F, x_flat)
        loss = (x_prime_flat * Fx).sum(dim=1)
        loss = loss * flat_flow_mask
        
        # loss = torch.einsum("bnc,bcc,bnc->bn", self.x_prime, estimated_F, self.x)
        # Fx = torch.matmul(estimated_F, torch.transpose(self.x, -1, -2))
        # Fx = torch.transpose(Fx, -1, -2)
        # loss = (self.x_prime * Fx).sum(dim=-1)
        # loss = loss * self.flow_mask
        
        return loss.abs().mean()

    def point_loss(self):

        P1 = torch.zeros((1, 3, 4), device=self.device)
        P1[:, :3, :3] = self.K.unsqueeze(1).unsqueeze(1)        # Unsqueezing to match H, W dimensions

        P2_norm = torch.zeros((self.num_frames - 2, 3, 4), device=self.device)
        P2_norm[:, :3, :3] = self.R[:-1]
        P2_norm[:, :3, 3] = self.trans[:-1]
        P2 = torch.matmul(self.K, P2_norm).unsqueeze(1).unsqueeze(1)

        P3_norm = torch.zeros((self.num_frames - 2, 3, 4), device=self.device)
        P3_norm[:, :3, :3] = self.R[1:]
        P3_norm[:, :3, 3] = self.trans[1:]
        P3 = torch.matmul(self.K, P3_norm).unsqueeze(1).unsqueeze(1)

        x_prime_unshift = torch.ones_like(self.x_prime)
        # clamping so that we do not index out of bounds:
        x_prime_unshift[..., 0] = torch.clamp(self.x_prime[..., 0] + self.W/2, 0, self.W-1)
        x_prime_unshift[..., 1] = torch.clamp(self.x_prime[..., 1] + self.H/2, 0, self.H-1)
        # indices => axis 0 is W, axis 1 is H

        indices = torch.round(x_prime_unshift).int()

        # next_flow = self.flow[1:].permute(0, 2, 1).reshape(self.flow.size(0)-1, 2, self.W, self.H)

        # batch_idx = torch.arange(1, self.num_frames - 1).reshape(-1, 1, 1).expand(-1, *indices.shape[1:-1])
        # next_flow1 = self.flow[batch_idx, indices[:-1, ..., 1], indices[:-1, ..., 0]]
        next_flow = torch.zeros_like(self.flow[1:])
        for f in range(self.num_frames - 2):
            i = indices[f, ..., 1].reshape(-1)
            j = indices[f, ..., 0].reshape(-1)
            next_flow[f, i, j] = self.flow[f+1, i, j]
        
        # [:,indices[:-1, :, 0], indices[:-1, :, 1],:]
        # TODO: Make sure the mask is multiplied to the loss
        
        next_corrs = self.homogenize(self.x_prime[:-1, ..., :-1] + next_flow)

        # correspondence is: (self.x, self.x_prime[:-1], next_corrs)

        x1_skew = self.skew_symmetric_matrix(self.x.expand(next_corrs.shape[0], -1, -1, -1))
        x2_skew = self.skew_symmetric_matrix(self.x_prime[:-1])
        x3_skew = self.skew_symmetric_matrix(next_corrs)

        constraint1 = torch.matmul(x1_skew[..., :-1, :], P1)
        constraint2 = torch.matmul(x2_skew[..., :-1, :], P2)
        constraint3 = torch.matmul(x3_skew[..., :-1, :], P3)

        constraint_matrix = torch.concat([constraint1, constraint2, constraint3], dim = -2)

        singular_values = torch.linalg.svdvals(constraint_matrix)

        loss = singular_values[..., -1]
        loss = loss * self.flow_mask[:-1] * self.flow_mask[1:]      # Mask out flows below threshold for both pairs

        return loss.mean()

    def loss(self, rl_wt = 0.01, pl_wt = 1.):

        # Rotation matrix:
        self.R = self.convert_rot_to_matrix(self.rots)

        rl = self.reproj_loss()
        # print(f"{rl.item()=}")
        # pl = self.point_loss()
        # print(f"{pl.item()=}")

        self.rl = rl.item()
        # self.pl = pl.item()

        # TODO: x_prime requires grad for some reason, figure out why and make it constant.

        # return pl * pl_wt + rl * rl_wt
        return rl
    
    def optimization_step(self):
        self.optimizer.zero_grad()

        # TODO: make sure grads propagate to right variables
        # TODO: convert to gpu memory after flow prediction

        loss = self.loss()
        loss.backward(retain_graph=True)

        # print(f"\r{loss.item()}", end="", flush=True)

        self.optimizer.step()
    
    @torch.no_grad()
    def visualize(self, images):

        # Rotation matrix:
        self.R = self.convert_rot_to_matrix(self.rots)      # (batch, 3, 3)

        P1 = torch.zeros((self.num_frames-1, 3, 4), device=self.device)
        P2 = torch.zeros((self.num_frames-1, 3, 4), device=self.device)

        P1[0, :3, :3] = torch.eye(3)
        P1[1:, :3, :3] = self.R[:-1]
        P1[1:, :3, 3] = self.trans[:-1]
        P1 = torch.matmul(self.K, P1).unsqueeze(1).unsqueeze(1)

        P2[:, :3, :3] = self.R[:]
        P2[:, :3, 3] = self.trans[:]
        P2 = torch.matmul(self.K, P2).unsqueeze(1).unsqueeze(1)     # Unsqueeze to accomodate height and width dimensions

        geometries = []

        for i in range(self.num_frames - 1):

            x1_skew = self.skew_symmetric_matrix(self.x[0])
            x2_skew = self.skew_symmetric_matrix(self.x_prime[i])

            constraint1 = torch.matmul(x1_skew[..., :-1, :], P1[i])
            constraint2 = torch.matmul(x2_skew[..., :-1, :], P2[i])

            constraint_matrix = torch.concat([constraint1, constraint2], dim = -2)

            U, S, Vh = torch.linalg.svd(constraint_matrix)
            X = Vh[..., -1, :]

            mask = self.flow_mask[i].bool()
            pts3d = X[mask]
            pts3d = pts3d / pts3d[:, -1, None]

            pcd = np.zeros((pts3d.shape[0], 3))
            colors = np.zeros((pts3d.shape[0], 3))

            pcd[:, :] = pts3d[:, :-1].cpu().detach().numpy()
            # all_pts.append(pcd)
            colors[:, :] = (images[i, :, mask] / 255.).cpu().numpy().T
            # all_colors.append(colors.T)

            pts_vis = o3d.geometry.PointCloud()
            pts_vis.points = o3d.utility.Vector3dVector(pcd)
            pts_vis.colors = o3d.utility.Vector3dVector(colors)

            geometries.append(pts_vis)

            # p1_camera = o3d.geometry.LineSet.create_camera_visualization(view_width_px=self.W, view_height_px=self.H, intrinsic=np.diag([self.focal, self.focal, 1.]), extrinsic=)
        
        T = np.eye(4)
        for i in range(self.num_frames):

            if i != 0:
                T[:3, :3] = self.R[i-1].cpu().detach().numpy()
                T[:3, 3] = self.trans[i-1].cpu().detach().numpy()

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
            frame.transform(T.copy())

            # text_position = T[:3, 3]  # Extract translation (camera position)
            # label_geometry = o3d.visualization.Text3D(
            #     text=f"Camera {i}",
            #     position=text_position,
            #     direction=(0, 0, -1),  # Text faces -Z direction
            #     up=(0, 1, 0),          # Up direction is Y
            #     font_size=20   # Scale the font size
            # )

            geometries.append(frame)
            # geometries.append(label_geometry)

        return geometries
    
    def visualize_sanity(self, images):

        # Rotation matrix:
        # self.R = self.convert_rot_to_matrix(self.rots)

        # P1 = torch.zeros((self.num_frames-1, 3, 4), device=self.device)
        # P2 = torch.zeros((self.num_frames-1, 3, 4), device=self.device)

        P1 = torch.tensor(np.load("sanity/P1.npy"), dtype=torch.float32).unsqueeze(0)
        P2 = torch.tensor(np.load("sanity/P2.npy"), dtype=torch.float32).unsqueeze(0)

        x = np.load("sanity/pts1.npy")
        x_prime = np.load("sanity/pts2.npy")

        x1_skew = self.skew_symmetric_matrix(self.homogenize(torch.tensor(x)))
        x2_skew = self.skew_symmetric_matrix(self.homogenize(torch.tensor(x_prime)))

        constraint1 = torch.matmul(x1_skew[..., :-1, :], P1)
        constraint2 = torch.matmul(x2_skew[..., :-1, :], P2)

        constraint_matrix = torch.concat([constraint1, constraint2], dim = -2)
        U, S, Vh = torch.linalg.svd(constraint_matrix)
        X = Vh[..., -1, :]

        pts3d = (X / X[:, -1, None])

        pcd = np.zeros((pts3d.shape[0], 3))

        pcd[:, :] = pts3d[:, :-1].cpu().numpy()

        
        pts_vis = o3d.geometry.PointCloud()
        pts_vis.points = o3d.utility.Vector3dVector(pcd)

            # p1_camera = o3d.geometry.LineSet.create_camera_visualization(view_width_px=self.W, view_height_px=self.H, intrinsic=np.diag([self.focal, self.focal, 1.]), extrinsic=)
        return [pts_vis]
    
def get_inliers(F, pts1, pts2, thresh=1e-10):

    pts1 = homogenize_2d(pts1)
    pts2 = homogenize_2d(pts2)

    l1 = pts1 @ F.T
    l2 = pts2 @ F

    d1 = np.abs((l1 * pts2).sum(axis=1)) / np.linalg.norm(l1[:, :2], axis=1)
    d2 = np.abs((l2 * pts1).sum(axis=1)) / np.linalg.norm(l2[:, :2], axis=1)

    return d1 + d2

    # prod = np.diag(pts2 @ F @ (pts1.T))
    # inliers = prod[np.abs(prod) < thresh]

    return inliers

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

def normalize_coordinates(pts):

    assert type(pts) == np.ndarray
    assert len(pts.shape) == 2
    assert pts.shape[1] == 2

    n = pts.shape[0]

    x = pts[:, 0]
    y = pts[:, 1]
    x0, y0 = np.mean(x), np.mean(y)

    d_avg = np.mean(np.sqrt((x - x0)**2 + (y - y0)**2))
    s = np.sqrt(2) / d_avg

    T = np.array([[s, 0., -s * x0],
                  [0., s, -s * y0],
                  [0., 0., 1.]])
    
    pts_hat = (T @ (homogenize_2d(pts).T)).T

    return pts_hat, T

def null_space(A, num=1):

    U, S, Vh = np.linalg.svd(A)
    return Vh[-num:, :]

def homogenize(v):
    ones = torch.ones((*v.shape[:-1], 1), device=v.device)
    return torch.cat((v, ones), dim=-1)

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
    
    
    return indices, correspondences, rgb

def compute_F_8pt(pts1, pts2):

    assert pts1.shape[0] >= 8 and pts2.shape[0] >= 8

    # Normalize coordinates:
    pts1_hat, T1 = normalize_coordinates(pts1)
    pts2_hat, T2 = normalize_coordinates(pts2)

    # Define A matrix for constraints on F:
    x1, y1, z1 = pts1_hat[:, 0], pts1_hat[:, 1], pts1_hat[:, 2]
    x2, y2, z2 = pts2_hat[:, 0], pts2_hat[:, 1], pts2_hat[:, 2]
    A = np.column_stack([x2*x1, x2*y1, x2*z1, y2*x1, y2*y1, y2*z1, z2*x1, z2*y1, z2*z1])

    # Recover normalized F from null space:
    f = null_space(A)
    F_hat = np.reshape(f, (3,3))

    # Project to rank 2 to constrain detF=0
    U, S, Vh = np.linalg.svd(F_hat)
    S[-1] = 0
    F_hat = U @ np.diag(S) @ Vh

    # Project back to pixel space:
    F = (T2.T) @ F_hat @ T1

    return F

def ransac_F(pts1, pts2, num_iters = 1000, thresh = 1e-3):
    num_inliers = 0
    best_F = None

    for i in tqdm(range(num_iters)):
        pt_indices = np.random.choice(pts1.shape[0], 8, replace=False)
        p1, p2 = pts1[pt_indices], pts2[pt_indices]

        F = compute_F_8pt(p1, p2)

        # inliers = get_inliers(F, pts1, pts2, thresh=thresh)
        errors = get_inliers(F, pts1, pts2, thresh=thresh)

        inliers = errors[errors < thresh]

        if len(inliers) > num_inliers:
            best_F = F
            num_inliers = len(inliers)

    inlier_percent = num_inliers / pts1.shape[0]
    
    return best_F, inlier_percent

def F_to_E(F):

    U, S, Vh = np.linalg.svd(F)

    S_ones = np.eye(3)
    S_ones[-1, -1] = 0

    E = U @ S_ones @ Vh

    return E

def F_to_P(F, v = np.array([0., 0., 0.]), lamb = 1.):

    U, S, Vh = np.linalg.svd(F.T)
    epipole = Vh[-1, :]

    ep_skew = skew_symmetric_matrix(epipole, mode="numpy")

    P = np.zeros((3, 4))
    P[:, :3] = ep_skew @ F + (epipole[:, np.newaxis] @ v[np.newaxis, :])
    P[:, 3] = lamb * epipole

    return P

def triangulate(P2, x, x_prime):
    P1 = torch.zeros((1, 3, 4), dtype=torch.float32, device = "cuda")
    P1[:, :3, :3] = torch.eye(3)

    P2 = torch.tensor(P2, dtype=torch.float32, device = "cuda").unsqueeze(0)

    x = torch.tensor(x, dtype=torch.float32, device = "cuda")
    x_prime = torch.tensor(x_prime, dtype=torch.float32, device = "cuda")

    x1_skew = skew_symmetric_matrix(homogenize(x))
    x2_skew = skew_symmetric_matrix(homogenize(x_prime))

    constraint1 = torch.matmul(x1_skew[..., :-1, :], P1)
    constraint2 = torch.matmul(x2_skew[..., :-1, :], P2)

    constraint_matrix = torch.concat([constraint1, constraint2], dim = -2)
    U, S, Vh = torch.linalg.svd(constraint_matrix)
    X = Vh[..., -1, :]

    pts3d = (X / X[:, -1, None])[:, :-1]

    return pts3d


def poses_from_E(E):

    W = np.array([[0, -1., 0.],
                  [1., 0., 0.],
                  [0., 0., 1.]])
    
    U, S, Vh = np.linalg.svd(E)

    C1 = U[:, -1]
    C2 = -U[:, -1]

    R1 = U @ W @ Vh
    R2 = U @ (W.T) @ Vh

    if np.linalg.det(R1) < 0:
        r1_correct = -1
    else:
        r1_correct = 1.

    if np.linalg.det(R2) < 0:
        r2_correct = -1
    else:
        r2_correct = 1.

    print(r1_correct, r2_correct)    

    return [(r1_correct * C1, r1_correct * R1), (r1_correct * C2, r1_correct * R1), 
            (r2_correct * C1, r2_correct * R2), (r2_correct * C2, r2_correct * R2)]

def visualize_poses(P2, x, x_prime, rgb = None):
    pts3d = triangulate(P2, x, x_prime)

    pcd = np.zeros((pts3d.shape[0], 3))
    pcd[:, :] = pts3d.cpu().detach().numpy()

    if rgb is not None:
        colors = np.zeros((pts3d.shape[0], 3))
        colors[:, :] = (rgb / 255.).cpu().detach().numpy()

    geometries = []
    pts_vis = o3d.geometry.PointCloud()
    pts_vis.points = o3d.utility.Vector3dVector(pcd)
    if rgb is not None:
        pts_vis.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pts_vis)

    # T = np.eye(4)
    # # T[:3, :3] = P2[0, :3, :3].cpu().detach().numpy()
    # # T[:3, 3] = P2[0, :3, 3].cpu().detach().numpy()
    # T[:3, :3] = P2[:3, :3]
    # T[:3, 3] = P2[:3, 3]

    # frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5., origin=[0, 0, 0])
    # frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5., origin=[0, 0, 0])
    # frame2.transform(T.copy())

    # geometries.append(frame1)
    # geometries.append(frame2)
    # geometries.append(label_geometry)

    return geometries

        # p1_camera = o3d.geometry.LineSet.create_camera_visualization(view_width_px=self.W, view_height_px=self.H, intrinsic=np.diag([self.focal, self.focal, 1.]), extrinsic=)
    # return [pts_vis]

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


    

