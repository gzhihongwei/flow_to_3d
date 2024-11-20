import torch
from decord import VideoReader
from sea_raft.raft import RAFT
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

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
        
        self.rots = nn.Parameter(torch.randn(self.num_frames-1, 2, 3))      # 6D rotation representation, (v1, v2)
        self.trans = nn.Parameter(torch.randn(self.num_frames-1, 3))
        self.focal = nn.Parameter(torch.abs(torch.randn(1)))

        # Convert focal length to intrinsics:
        self.K = torch.eye(3, device=self.device).unsqueeze(0)
        self.K[0, 0, 0] = torch.abs(self.focal)
        self.K[0, 1, 1] = torch.abs(self.focal)
        self.Kt = torch.transpose(self.K, -1, -2)
        self.Kt_inv = torch.transpose(torch.inverse(self.K), -1, -2)

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
        
        vv, uu = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")

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

    def loss(self, rl_wt = 0.001, pl_wt = 1.):

        # Rotation matrix:
        self.R = self.convert_rot_to_matrix(self.rots)

        rl = self.reproj_loss()
        pl = self.point_loss()

        # TODO: x_prime requires grad for some reason, figure out why and make it constant.

        return pl * pl_wt + rl * rl_wt
    
    def optimization_step(self):
        self.optimizer.zero_grad()

        # TODO: make sure grads propagate to right variables
        # TODO: convert to gpu memory after flow prediction

        loss = self.loss()
        loss.backward()

        self.optimizer.step()

    

