import argparse
from pathlib import Path

from config.parser import parse_args
from flow_utils import read_video, load_flow_model, create_flow_heatmap, create_flow_mask_heatmap, create_flow_correspondence_video

def main(args):

    video = read_video(args)          # video is of shape (frames, H, W, 3)
    video = video[10:22]  # If you only want a certain subset of frames

    model = load_flow_model(args)
    create_flow_heatmap(video, model, args)
    create_flow_mask_heatmap(video, model, args)
    create_flow_correspondence_video(video, model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', default=Path(__file__).resolve().parent / "config/eval/spring-M.json", type=Path)
    parser.add_argument('--url', help='checkpoint url', type=str, default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M")
    parser.add_argument('--device', help='inference device', type=str, default='cuda')
    parser.add_argument('--video_path', help='relative file path of the video', type=str, required=True)
    parser.add_argument('--batch', help='Number of frames to batch compute flow', type=int, required=True)
    args = parse_args(parser)
    
    main(args)