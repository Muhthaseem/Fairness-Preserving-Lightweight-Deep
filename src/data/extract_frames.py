"""
Frame extraction from video datasets (FF++, Celeb-DF, DFD).
Extracts frames uniformly from each video and saves as JPEG.
"""
import os
import cv2
import glob
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import (
    FF_ROOT, FF_ORIGINAL, FF_FAKE_TYPES,
    CELEBDF_ROOT, CELEBDF_REAL, CELEBDF_SYNTHESIS, CELEBDF_YOUTUBE,
    DFD_ROOT, DFD_MANIPULATED, DFD_ORIGINAL,
    FRAMES_DIR, FRAMES_PER_VIDEO
)


def extract_frames_from_video(video_path, output_dir, num_frames=FRAMES_PER_VIDEO,
                                quality=95):
    """
    Extract num_frames uniformly spaced frames from a video.

    Args:
        video_path: Path to the video file (.mp4)
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract per video
        quality: JPEG quality (1-100)

    Returns:
        List of saved frame paths, or empty list on failure
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[WARN] Could not open: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Calculate frame indices to extract (uniformly spaced)
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / num_frames
        frame_indices = [int(i * step) for i in range(num_frames)]

    saved_paths = []
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_filename = f"{video_name}_frame{idx:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        saved_paths.append(frame_path)

    cap.release()
    return saved_paths


def extract_ff_frames(num_frames=FRAMES_PER_VIDEO, max_workers=4):
    """
    Extract frames from all FaceForensics++ videos.

    Processes:
        - original/ -> REAL labels
        - Deepfakes/, Face2Face/, FaceSwap/, NeuralTextures/, FaceShifter/ -> FAKE labels

    Returns:
        DataFrame with columns: [frame_path, video_path, label, source, video_id]
    """
    records = []

    # Process original (REAL) videos
    real_dir = FF_ORIGINAL
    videos = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))
    print(f"\n[FF++] Extracting frames from {len(videos)} REAL videos...")

    out_dir = os.path.join(FRAMES_DIR, "FF++", "original")
    for video_path in tqdm(videos, desc="FF++ original"):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        frames = extract_frames_from_video(video_path, out_dir, num_frames)
        for fp in frames:
            records.append({
                "frame_path": fp,
                "video_path": video_path,
                "label": "REAL",
                "source": "FF++_original",
                "video_id": f"original_{video_id}",
                "dataset": "FF++"
            })

    # Process each fake type
    for fake_type in FF_FAKE_TYPES:
        fake_dir = os.path.join(FF_ROOT, fake_type)
        if not os.path.exists(fake_dir):
            print(f"[WARN] Fake type dir not found: {fake_dir}")
            continue

        videos = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))
        print(f"\n[FF++] Extracting frames from {len(videos)} {fake_type} videos...")

        out_dir = os.path.join(FRAMES_DIR, "FF++", fake_type)
        for video_path in tqdm(videos, desc=f"FF++ {fake_type}"):
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            frames = extract_frames_from_video(video_path, out_dir, num_frames)
            for fp in frames:
                records.append({
                    "frame_path": fp,
                    "video_path": video_path,
                    "label": "FAKE",
                    "source": f"FF++_{fake_type}",
                    "video_id": f"{fake_type}_{video_id}",
                    "dataset": "FF++"
                })

    # Process DeepFakeDetection separately
    dfd_ff_dir = os.path.join(FF_ROOT, "DeepFakeDetection")
    if os.path.exists(dfd_ff_dir):
        videos = sorted(glob.glob(os.path.join(dfd_ff_dir, "*.mp4")))
        print(f"\n[FF++] Extracting frames from {len(videos)} DeepFakeDetection videos...")
        out_dir = os.path.join(FRAMES_DIR, "FF++", "DeepFakeDetection")
        for video_path in tqdm(videos, desc="FF++ DeepFakeDetection"):
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            frames = extract_frames_from_video(video_path, out_dir, num_frames)
            for fp in frames:
                records.append({
                    "frame_path": fp,
                    "video_path": video_path,
                    "label": "FAKE",
                    "source": "FF++_DeepFakeDetection",
                    "video_id": f"DFD_{video_id}",
                    "dataset": "FF++"
                })

    df = pd.DataFrame(records)
    csv_path = os.path.join(FRAMES_DIR, "ff_frames.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[FF++] Total frames extracted: {len(df)}")
    print(f"[FF++] REAL: {(df['label']=='REAL').sum()}, FAKE: {(df['label']=='FAKE').sum()}")
    print(f"[FF++] Saved manifest to: {csv_path}")
    return df


def extract_celebdf_frames(num_frames=FRAMES_PER_VIDEO):
    """
    Extract frames from Celeb-DF dataset.

    Processes:
        - Celeb-real/ + YouTube-real/ -> REAL
        - Celeb-synthesis/ -> FAKE

    Returns:
        DataFrame with columns: [frame_path, video_path, label, source, video_id]
    """
    records = []

    # Process real videos
    for real_name, real_dir in [("Celeb-real", CELEBDF_REAL), ("YouTube-real", CELEBDF_YOUTUBE)]:
        if not os.path.exists(real_dir):
            continue
        videos = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))
        print(f"\n[Celeb-DF] Extracting frames from {len(videos)} {real_name} videos...")

        out_dir = os.path.join(FRAMES_DIR, "Celeb-DF", real_name)
        for video_path in tqdm(videos, desc=f"Celeb-DF {real_name}"):
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            frames = extract_frames_from_video(video_path, out_dir, num_frames)
            for fp in frames:
                records.append({
                    "frame_path": fp,
                    "video_path": video_path,
                    "label": "REAL",
                    "source": f"CelebDF_{real_name}",
                    "video_id": f"{real_name}_{video_id}",
                    "dataset": "Celeb-DF"
                })

    # Process fake videos
    if os.path.exists(CELEBDF_SYNTHESIS):
        videos = sorted(glob.glob(os.path.join(CELEBDF_SYNTHESIS, "*.mp4")))
        print(f"\n[Celeb-DF] Extracting frames from {len(videos)} synthesis videos...")

        out_dir = os.path.join(FRAMES_DIR, "Celeb-DF", "Celeb-synthesis")
        for video_path in tqdm(videos, desc="Celeb-DF synthesis"):
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            frames = extract_frames_from_video(video_path, out_dir, num_frames)
            for fp in frames:
                records.append({
                    "frame_path": fp,
                    "video_path": video_path,
                    "label": "FAKE",
                    "source": "CelebDF_synthesis",
                    "video_id": f"synthesis_{video_id}",
                    "dataset": "Celeb-DF"
                })

    df = pd.DataFrame(records)
    csv_path = os.path.join(FRAMES_DIR, "celebdf_frames.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Celeb-DF] Total frames extracted: {len(df)}")
    print(f"[Celeb-DF] REAL: {(df['label']=='REAL').sum()}, FAKE: {(df['label']=='FAKE').sum()}")
    print(f"[Celeb-DF] Saved manifest to: {csv_path}")
    return df


def extract_dfd_frames(num_frames=FRAMES_PER_VIDEO):
    """
    Extract frames from Google DeepFake Detection (DFD) dataset.

    Returns:
        DataFrame with columns: [frame_path, video_path, label, source, video_id]
    """
    records = []

    # Original (real) sequences
    if os.path.exists(DFD_ORIGINAL):
        videos = sorted(glob.glob(os.path.join(DFD_ORIGINAL, "*.mp4")))
        # Also check subdirectories
        if not videos:
            videos = sorted(glob.glob(os.path.join(DFD_ORIGINAL, "**", "*.mp4"), recursive=True))
        print(f"\n[DFD] Extracting frames from {len(videos)} original videos...")

        out_dir = os.path.join(FRAMES_DIR, "DFD", "original")
        for video_path in tqdm(videos, desc="DFD original"):
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            frames = extract_frames_from_video(video_path, out_dir, num_frames)
            for fp in frames:
                records.append({
                    "frame_path": fp,
                    "video_path": video_path,
                    "label": "REAL",
                    "source": "DFD_original",
                    "video_id": f"DFD_orig_{video_id}",
                    "dataset": "DFD"
                })

    # Manipulated sequences
    if os.path.exists(DFD_MANIPULATED):
        videos = sorted(glob.glob(os.path.join(DFD_MANIPULATED, "*.mp4")))
        if not videos:
            videos = sorted(glob.glob(os.path.join(DFD_MANIPULATED, "**", "*.mp4"), recursive=True))
        print(f"\n[DFD] Extracting frames from {len(videos)} manipulated videos...")

        out_dir = os.path.join(FRAMES_DIR, "DFD", "manipulated")
        for video_path in tqdm(videos, desc="DFD manipulated"):
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            frames = extract_frames_from_video(video_path, out_dir, num_frames)
            for fp in frames:
                records.append({
                    "frame_path": fp,
                    "video_path": video_path,
                    "label": "FAKE",
                    "source": "DFD_manipulated",
                    "video_id": f"DFD_manip_{video_id}",
                    "dataset": "DFD"
                })

    df = pd.DataFrame(records)
    csv_path = os.path.join(FRAMES_DIR, "dfd_frames.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[DFD] Total frames extracted: {len(df)}")
    print(f"[DFD] REAL: {(df['label']=='REAL').sum()}, FAKE: {(df['label']=='FAKE').sum()}")
    print(f"[DFD] Saved manifest to: {csv_path}")
    return df


def extract_all_frames(num_frames=FRAMES_PER_VIDEO):
    """Extract frames from all three datasets."""
    print("=" * 60)
    print("FRAME EXTRACTION PIPELINE")
    print("=" * 60)

    ff_df = extract_ff_frames(num_frames)
    celebdf_df = extract_celebdf_frames(num_frames)
    dfd_df = extract_dfd_frames(num_frames)

    # Combine all
    all_df = pd.concat([ff_df, celebdf_df, dfd_df], ignore_index=True)
    all_csv = os.path.join(FRAMES_DIR, "all_frames.csv")
    all_df.to_csv(all_csv, index=False)

    print("\n" + "=" * 60)
    print(f"TOTAL FRAMES EXTRACTED: {len(all_df)}")
    print(f"  FF++:     {len(ff_df)}")
    print(f"  Celeb-DF: {len(celebdf_df)}")
    print(f"  DFD:      {len(dfd_df)}")
    print(f"Combined manifest saved to: {all_csv}")
    print("=" * 60)

    return all_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from video datasets")
    parser.add_argument("--num_frames", type=int, default=FRAMES_PER_VIDEO,
                        help=f"Frames to extract per video (default: {FRAMES_PER_VIDEO})")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "ff", "celebdf", "dfd"],
                        help="Which dataset to process")
    args = parser.parse_args()

    if args.dataset == "all":
        extract_all_frames(args.num_frames)
    elif args.dataset == "ff":
        extract_ff_frames(args.num_frames)
    elif args.dataset == "celebdf":
        extract_celebdf_frames(args.num_frames)
    elif args.dataset == "dfd":
        extract_dfd_frames(args.num_frames)
