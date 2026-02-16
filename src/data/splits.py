"""
Dataset splitting: stratified train/val/test by video ID and demographic group.
Prevents data leakage (no frames from the same video in different splits).
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from collections import Counter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import (
    FACES_DIR, SPLITS_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    DEMOGRAPHIC_GROUPS, NUM_GROUPS
)


def create_splits(annotated_csv, output_dir=SPLITS_DIR, seed=RANDOM_SEED):
    """
    Create train/val/test splits stratified by label and demographic group.
    Splits are done at the VIDEO level to prevent data leakage.

    Args:
        annotated_csv: Path to annotated faces CSV
        output_dir: Directory to save split CSVs
        seed: Random seed

    Returns:
        dict: {'train': DataFrame, 'val': DataFrame, 'test': DataFrame}
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(annotated_csv)
    print(f"Total samples: {len(df)}")
    print(f"Unique videos: {df['video_id'].nunique()}")

    # Create a stratification key combining label and demographic group
    df["strat_key"] = df["label"] + "_" + df["demographic_group"]

    # Get unique videos with their stratification info
    video_info = df.groupby("video_id").agg({
        "strat_key": "first",  # Use first frame's group as video's group
        "label": "first",
        "demographic_group": "first",
        "group_id": "first",
    }).reset_index()

    # First split: separate test set (15%)
    # Stratify by label + demographic group
    try:
        train_val_videos, test_videos = train_test_split(
            video_info,
            test_size=TEST_RATIO,
            stratify=video_info["strat_key"],
            random_state=seed
        )
    except ValueError:
        # If some strat groups are too small, fall back to label-only stratification
        print("[WARN] Some demographic groups too small for full stratification. "
              "Falling back to label-only stratification.")
        train_val_videos, test_videos = train_test_split(
            video_info,
            test_size=TEST_RATIO,
            stratify=video_info["label"],
            random_state=seed
        )

    # Second split: separate validation from training (15% of total = ~17.6% of remaining)
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    try:
        train_videos, val_videos = train_test_split(
            train_val_videos,
            test_size=val_ratio_adjusted,
            stratify=train_val_videos["strat_key"],
            random_state=seed
        )
    except ValueError:
        train_videos, val_videos = train_test_split(
            train_val_videos,
            test_size=val_ratio_adjusted,
            stratify=train_val_videos["label"],
            random_state=seed
        )

    # Map videos back to frames
    train_video_ids = set(train_videos["video_id"])
    val_video_ids = set(val_videos["video_id"])
    test_video_ids = set(test_videos["video_id"])

    train_df = df[df["video_id"].isin(train_video_ids)].copy()
    val_df = df[df["video_id"].isin(val_video_ids)].copy()
    test_df = df[df["video_id"].isin(test_video_ids)].copy()

    # Save splits
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Print split statistics
    print("\n" + "=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{split_name}: {len(split_df)} samples, "
              f"{split_df['video_id'].nunique()} videos")
        print(f"  REAL: {(split_df['label']=='REAL').sum()}, "
              f"FAKE: {(split_df['label']=='FAKE').sum()}")
        print(f"  Demographic distribution:")
        for group_name in DEMOGRAPHIC_GROUPS:
            count = (split_df["demographic_group"] == group_name).sum()
            pct = count / len(split_df) * 100 if len(split_df) > 0 else 0
            print(f"    {group_name:20s}: {count:6d} ({pct:5.1f}%)")

    # Verify no video leakage
    assert len(train_video_ids & val_video_ids) == 0, "Video leakage: train & val"
    assert len(train_video_ids & test_video_ids) == 0, "Video leakage: train & test"
    assert len(val_video_ids & test_video_ids) == 0, "Video leakage: val & test"
    print("\n✓ No video leakage between splits.")

    print(f"\nSaved splits to: {output_dir}")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

    return {"train": train_df, "val": val_df, "test": test_df}


def create_cross_dataset_splits(output_dir=SPLITS_DIR):
    """
    Create test-only splits for cross-dataset evaluation (Celeb-DF, DFD).
    These datasets are used purely for testing — never for training.
    """
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, manifest_name in [
        ("Celeb-DF", "celebdf_faces_annotated.csv"),
        ("DFD", "dfd_faces_annotated.csv"),
    ]:
        manifest_path = os.path.join(FACES_DIR, manifest_name)
        if not os.path.exists(manifest_path):
            print(f"[WARN] Cross-dataset manifest not found: {manifest_path}")
            continue

        df = pd.read_csv(manifest_path)
        out_path = os.path.join(output_dir, f"{dataset_name}_test.csv")
        df.to_csv(out_path, index=False)

        print(f"\n{dataset_name} test set: {len(df)} samples")
        print(f"  REAL: {(df['label']=='REAL').sum()}, "
              f"FAKE: {(df['label']=='FAKE').sum()}")
        for group_name in DEMOGRAPHIC_GROUPS:
            count = (df["demographic_group"] == group_name).sum()
            pct = count / len(df) * 100 if len(df) > 0 else 0
            print(f"    {group_name:20s}: {count:6d} ({pct:5.1f}%)")

        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create dataset splits")
    parser.add_argument("--annotated_csv", type=str,
                        default=os.path.join(FACES_DIR, "ff_faces_annotated.csv"),
                        help="Path to annotated faces CSV (FF++ for primary splits)")
    args = parser.parse_args()

    # Create primary FF++ splits
    create_splits(args.annotated_csv)

    # Create cross-dataset test sets
    create_cross_dataset_splits()
