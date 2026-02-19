"""
Script 03: Annotate faces with demographic predictions (race, gender).
Usage: python scripts/03_annotate_demographics.py [--model_path path/to/fairface.pth]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.demographic_annotator import annotate_all_faces


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to FairFace model weights")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for annotation")
    args = parser.parse_args()
    annotate_all_faces(model_path=args.model_path, batch_size=args.batch_size)
