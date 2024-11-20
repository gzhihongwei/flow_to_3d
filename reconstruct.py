import torch
import decord
decord.bridge.set_bridge("torch")
import argparse
from config.parser import parse_args
import torch.nn.functional as F

from decord import VideoReader

from sea_raft.raft import RAFT
from views import Views


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
    tensor = torch.tensor([1]).cuda()
    video = read_video_decord(video).permute(0, 3, 1, 2)#.to(args.device)
    image1 = video[20:25]
    image2 = video[21:26]
    flow, _ = calc_flow(args, model, image1, image2)
    flow = flow.permute(0, 2, 3, 1)
    views = Views(args, flow)
    views = views.to(args.device)
    views.optimization_step()

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
    # model = model.to(args.device)
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