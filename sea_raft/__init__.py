import argparse

import torch
from config.parser import parse_args

from pathlib import Path

from .raft import RAFT

def load_RAFT():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
    parser.add_argument('--device', help='inference device', type=str, default='cpu')
    args = parse_args(parser, ["--cfg", str(Path(__file__).resolve().parent.parent / "config/eval/spring-M.json"), "--url", "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"])
    
    model = RAFT.from_pretrained(args.url, args=args)
        
    if args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    return model, args
