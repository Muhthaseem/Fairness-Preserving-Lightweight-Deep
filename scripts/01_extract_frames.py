"""
Script 01: Extract frames from all video datasets.
Usage: python scripts/01_extract_frames.py [--num_frames 30] [--dataset all]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.extract_frames import extract_all_frames, extract_ff_frames, \
    extract_celebdf_frames, extract_dfd_frames
from src.config import FRAMES_PER_VIDEO


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=FRAMES_PER_VIDEO)
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "ff", "celebdf", "dfd"])
    args = parser.parse_args()

    if args.dataset == "all":
        extract_all_frames(args.num_frames)
    elif args.dataset == "ff":
        extract_ff_frames(args.num_frames)
    elif args.dataset == "celebdf":
        extract_celebdf_frames(args.num_frames)
    elif args.dataset == "dfd":
        extract_dfd_frames(args.num_frames)
