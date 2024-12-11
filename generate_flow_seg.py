import argparse
from config.parser import parse_args

from flow_utils import read_video, load_flow_model, predict_flow, create_flow_heatmap, create_flow_mask_heatmap, create_flow_correspondence_video

def main(args):

    video = read_video(args)          # video is of shape (frames, H, W, 3)
    video = video[10:22]

    model = load_flow_model(args)
    create_flow_heatmap(video, model, args)
    create_flow_mask_heatmap(video, model, args)
    create_flow_correspondence_video(video, model, args)
    print("")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')
    parser.add_argument('--video_path', help='relative file path of the video', type=str, default=None)
    parser.add_argument('--thresh', help='relative file path of the video', type=float, default=None)
    parser.add_argument('--batch', help='relative file path of the video', type=int, default=None)
    parser.add_argument('--skip', help='relative file path of the video', type=int, default=None)
    args = parse_args(parser)
    
    main(args)