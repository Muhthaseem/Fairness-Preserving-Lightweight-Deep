"""
Script 04: Create stratified train/val/test splits.
Usage: python scripts/04_create_splits.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.splits import create_splits, create_cross_dataset_splits
from src.config import FACES_DIR


if __name__ == "__main__":
    # Create primary FF++ splits
    ff_annotated = os.path.join(FACES_DIR, "ff_faces_annotated.csv")
    if os.path.exists(ff_annotated):
        create_splits(ff_annotated)
    else:
        # Try combined manifest
        all_annotated = os.path.join(FACES_DIR, "all_faces_annotated.csv")
        if os.path.exists(all_annotated):
            create_splits(all_annotated)
        else:
            print("ERROR: No annotated faces manifest found. Run scripts 01-03 first.")

    # Create cross-dataset test sets
    create_cross_dataset_splits()
