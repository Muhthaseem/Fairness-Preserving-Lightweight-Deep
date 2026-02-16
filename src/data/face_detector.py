"""
Face detection and cropping using MTCNN.
Crops and aligns faces to 256x256 for all extracted frames.
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import (
    IMAGE_SIZE, FACES_DIR, FRAMES_DIR,
    FACE_DETECTION_BATCH_SIZE, MIN_FACE_SIZE, DEVICE
)


class FaceDetector:
    """MTCNN-based face detector for extracting and aligning faces."""

    def __init__(self, image_size=IMAGE_SIZE, min_face_size=MIN_FACE_SIZE,
                 device=DEVICE):
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=40,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,  # Return PIL Image, not normalized tensor
            select_largest=True,  # Select largest face per frame
            device=device,
        )
        self.image_size = image_size

    def detect_and_crop(self, image_path):
        """
        Detect face in an image and return cropped face.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image of the cropped face, or None if no face detected
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open image: {image_path}: {e}")
            return None

        # Detect faces
        face = self.mtcnn(img)
        if face is None:
            return None

        # face is a tensor of shape (3, image_size, image_size), values in [0, 255]
        if isinstance(face, torch.Tensor):
            face_np = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            face_pil = Image.fromarray(face_np)
            return face_pil
        return face

    def detect_and_crop_batch(self, image_paths):
        """
        Batch face detection.

        Args:
            image_paths: List of image paths

        Returns:
            List of (face_pil, image_path) tuples for successfully detected faces
        """
        images = []
        valid_paths = []

        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception:
                continue

        if not images:
            return []

        # Batch detect
        faces = self.mtcnn(images)
        results = []
        for face, path in zip(faces, valid_paths):
            if face is not None:
                if isinstance(face, torch.Tensor):
                    face_np = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    face_pil = Image.fromarray(face_np)
                    results.append((face_pil, path))
        return results


def process_faces_from_manifest(frames_csv, output_base_dir=FACES_DIR,
                                 batch_size=FACE_DETECTION_BATCH_SIZE):
    """
    Process all frames from a manifest CSV and extract faces.

    Args:
        frames_csv: Path to frames CSV (from extract_frames.py)
        output_base_dir: Base directory to save cropped faces
        batch_size: Batch size for MTCNN processing

    Returns:
        DataFrame with face paths added
    """
    df = pd.read_csv(frames_csv)
    detector = FaceDetector()

    face_records = []
    failed_count = 0

    # Process in batches
    all_paths = df["frame_path"].tolist()
    total_batches = (len(all_paths) + batch_size - 1) // batch_size

    print(f"\nProcessing {len(all_paths)} frames in {total_batches} batches...")

    for batch_start in tqdm(range(0, len(all_paths), batch_size),
                            total=total_batches, desc="Face detection"):
        batch_paths = all_paths[batch_start:batch_start + batch_size]
        batch_rows = df.iloc[batch_start:batch_start + batch_size]

        for i, frame_path in enumerate(batch_paths):
            row = batch_rows.iloc[i]
            face = detector.detect_and_crop(frame_path)

            if face is None:
                failed_count += 1
                continue

            # Construct output path mirroring the frame structure
            rel_path = os.path.relpath(frame_path, FRAMES_DIR)
            face_path = os.path.join(output_base_dir, rel_path)
            os.makedirs(os.path.dirname(face_path), exist_ok=True)

            # Save as JPEG
            face.save(face_path, quality=95)

            face_records.append({
                "face_path": face_path,
                "frame_path": frame_path,
                "video_path": row["video_path"],
                "label": row["label"],
                "source": row["source"],
                "video_id": row["video_id"],
                "dataset": row["dataset"],
            })

    face_df = pd.DataFrame(face_records)
    csv_path = frames_csv.replace("frames.csv", "faces.csv").replace(
        FRAMES_DIR, FACES_DIR
    )
    # Save in faces dir
    csv_out = os.path.join(FACES_DIR, os.path.basename(frames_csv).replace(
        "frames", "faces"))
    face_df.to_csv(csv_out, index=False)

    detection_rate = len(face_records) / len(all_paths) * 100
    print(f"\nFace detection complete:")
    print(f"  Total frames:   {len(all_paths)}")
    print(f"  Faces detected: {len(face_records)} ({detection_rate:.1f}%)")
    print(f"  Failed:         {failed_count}")
    print(f"  Saved manifest: {csv_out}")

    return face_df


def process_all_faces():
    """Process faces from all dataset frame manifests."""
    print("=" * 60)
    print("FACE DETECTION PIPELINE")
    print("=" * 60)

    all_dfs = []

    for manifest_name in ["ff_frames.csv", "celebdf_frames.csv", "dfd_frames.csv"]:
        manifest_path = os.path.join(FRAMES_DIR, manifest_name)
        if os.path.exists(manifest_path):
            print(f"\nProcessing: {manifest_name}")
            face_df = process_faces_from_manifest(manifest_path)
            all_dfs.append(face_df)
        else:
            print(f"[WARN] Manifest not found: {manifest_path}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(FACES_DIR, "all_faces.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n{'=' * 60}")
        print(f"TOTAL FACES: {len(combined)}")
        print(f"Combined manifest: {combined_path}")
        print(f"{'=' * 60}")
        return combined

    return pd.DataFrame()


if __name__ == "__main__":
    process_all_faces()
