"""
Demographic annotation using FairFace model.
Predicts race and gender for each cropped face image.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import (
    FACES_DIR, DEVICE, FAIRFACE_RACE_MAP,
    GROUP_TO_IDX, DEMOGRAPHIC_GROUPS, MODELS_DIR
)


# FairFace model labels
FAIRFACE_RACES = ["White", "Black", "Latino_Hispanic", "East Asian",
                  "Southeast Asian", "Indian", "Middle Eastern"]
FAIRFACE_GENDERS = ["Male", "Female"]


class FairFaceAnnotator:
    """
    Demographic annotator using a FairFace-style model.
    Predicts race (7 classes) and gender (2 classes) for face images.
    """

    def __init__(self, model_path=None, device=DEVICE):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Initialize model structure (ResNet34)
        self.model = models.resnet34(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 18) # FairFace has 18 outputs (Race+Gender+Age) usually, but here we strictly need 7+2
        
        # Standard FairFace (7 race, 2 gender) structure usually is one model or two. 
        # Here we use the specific architecture expected by 'res34_fair_align_multi_7_20190809.pt'
        # which is a standard ResNet34. The official weights are full state dicts.
        
        # Official FairFace model weights on Google Drive
        # ID: 11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB
        file_id = "11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB"
        save_path = os.path.join(MODELS_DIR, "res34_fair_align_multi_7_20190809.pt")

        if not os.path.exists(save_path):
             print(f"Downloading FairFace weights to {save_path}...")
             try:
                 self._download_file_from_google_drive(file_id, save_path)
                 if os.path.exists(save_path) and os.path.getsize(save_path) < 1000:
                     # If file is too small, it's likely an error page
                     print("Download file too small, likely failed.")
                     os.remove(save_path)
             except Exception as e:
                 print(f"Failed to download weights: {e}")

        # Re-build for specific heads as used in our project
        self.model = models.resnet34(weights=None) # Reset
        self.model.fc = nn.Identity()
        self.race_head = nn.Linear(num_features, 7)
        self.gender_head = nn.Linear(num_features, 2)
        
        if os.path.exists(save_path):
            state_dict = torch.load(save_path, map_location=device, weights_only=False)
            # The official model output is a single linear layer of size 9 (7 race + 2 gender) OR 18.
            # We need to be careful. The official release is `fc` layer with 9 outputs.
            # 0-6: Race, 7-8: Gender
            
            # Load backbone
            self.model.load_state_dict(state_dict, strict=False)
            
            # Extract weights for heads manually from 'fc.weight' and 'fc.bias' of the loaded state_dict
            if 'fc.weight' in state_dict:
                # Race (first 7)
                self.race_head.weight.data = state_dict['fc.weight'][:7]
                self.race_head.bias.data = state_dict['fc.bias'][:7]
                # Gender (next 2) -> actually indices 7 and 8
                self.gender_head.weight.data = state_dict['fc.weight'][7:9]
                self.gender_head.bias.data = state_dict['fc.bias'][7:9]
                print(f"[INFO] Loaded FairFace weights from {save_path}")
        else:
             print("[WARN] Using random weights! Results will be garbage.")

        self.model = self.model.to(device)
        self.race_head = self.race_head.to(device)
        self.gender_head = self.gender_head.to(device)
        
        self.model.eval()
        self.race_head.eval()
        self.gender_head.eval()

    def _download_file_from_google_drive(self, id, destination):
        import requests
        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params={'id': id}, stream=True)
        token = self._get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        self._save_response_content(response, destination)

    def _get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def _save_response_content(self, response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    @torch.no_grad()
    def predict(self, image):
        """
        Predict race and gender for a single face image.

        Args:
            image: PIL Image or path to image

        Returns:
            dict with keys: race, gender, race_conf, gender_conf, demographic_group, group_id
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        x = self.transform(image).unsqueeze(0).to(self.device)
        features = self.model(x)

        race_logits = self.race_head(features)
        gender_logits = self.gender_head(features)

        race_probs = torch.softmax(race_logits, dim=1)
        gender_probs = torch.softmax(gender_logits, dim=1)

        race_idx = race_probs.argmax(dim=1).item()
        gender_idx = gender_probs.argmax(dim=1).item()

        race_ff = FAIRFACE_RACES[race_idx]
        gender = FAIRFACE_GENDERS[gender_idx]

        # Map FairFace race to our 4 simplified categories
        race = FAIRFACE_RACE_MAP.get(race_ff, "Other")

        demographic_group = f"{gender}-{race}"
        group_id = GROUP_TO_IDX.get(demographic_group, -1)

        return {
            "race": race,
            "race_detailed": race_ff,
            "gender": gender,
            "race_confidence": race_probs.max().item(),
            "gender_confidence": gender_probs.max().item(),
            "demographic_group": demographic_group,
            "group_id": group_id,
        }

    @torch.no_grad()
    def predict_batch(self, images):
        """
        Batch prediction of race and gender.

        Args:
            images: List of PIL Images or paths

        Returns:
            List of prediction dicts
        """
        batch_tensors = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            batch_tensors.append(self.transform(img))

        batch = torch.stack(batch_tensors).to(self.device)
        features = self.model(batch)

        race_logits = self.race_head(features)
        gender_logits = self.gender_head(features)

        race_probs = torch.softmax(race_logits, dim=1)
        gender_probs = torch.softmax(gender_logits, dim=1)

        results = []
        for i in range(len(images)):
            race_idx = race_probs[i].argmax().item()
            gender_idx = gender_probs[i].argmax().item()

            race_ff = FAIRFACE_RACES[race_idx]
            gender = FAIRFACE_GENDERS[gender_idx]
            race = FAIRFACE_RACE_MAP.get(race_ff, "Other")
            demographic_group = f"{gender}-{race}"
            group_id = GROUP_TO_IDX.get(demographic_group, -1)

            results.append({
                "race": race,
                "race_detailed": race_ff,
                "gender": gender,
                "race_confidence": race_probs[i].max().item(),
                "gender_confidence": gender_probs[i].max().item(),
                "demographic_group": demographic_group,
                "group_id": group_id,
            })

        return results


def annotate_faces_from_manifest(faces_csv, batch_size=32, model_path=None):
    """
    Annotate all faces in a manifest CSV with demographic predictions.

    Args:
        faces_csv: Path to faces CSV
        batch_size: Batch size for prediction
        model_path: Optional path to FairFace weights

    Returns:
        DataFrame with demographic columns added
    """
    df = pd.read_csv(faces_csv)
    annotator = FairFaceAnnotator(model_path=model_path)

    # Prepare result columns
    races = []
    genders = []
    groups = []
    group_ids = []
    race_confs = []
    gender_confs = []

    all_paths = df["face_path"].tolist()
    total_batches = (len(all_paths) + batch_size - 1) // batch_size

    print(f"\nAnnotating {len(all_paths)} faces in {total_batches} batches...")

    for batch_start in tqdm(range(0, len(all_paths), batch_size),
                            total=total_batches, desc="Demographic annotation"):
        batch_paths = all_paths[batch_start:batch_start + batch_size]

        # Load and predict batch
        valid_images = []
        valid_indices = []
        # Load batch images in parallel
        from concurrent.futures import ThreadPoolExecutor

        def load_img(idx_path):
            i, path = idx_path
            try:
                return i, Image.open(path).convert("RGB")
            except Exception:
                return i, None

        valid_images = []
        valid_indices = []
        
        # Prepare args for map (enumerate logic)
        args = [(i, path) for i, path in enumerate(batch_paths)]

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(load_img, args))
        
        # Sort results to maintain order (though map usually preserves it)
        # and filter Nones
        for local_len_idx, img in results:
            if img is not None:
                valid_images.append(img)
                valid_indices.append(batch_start + local_len_idx)

        if valid_images:
            preds = annotator.predict_batch(valid_images)
        else:
            preds = []

        # Fill predictions for this batch
        pred_idx = 0
        for i in range(len(batch_paths)):
            global_idx = batch_start + i
            if global_idx in valid_indices and pred_idx < len(preds):
                p = preds[pred_idx]
                races.append(p["race"])
                genders.append(p["gender"])
                groups.append(p["demographic_group"])
                group_ids.append(p["group_id"])
                race_confs.append(p["race_confidence"])
                gender_confs.append(p["gender_confidence"])
                pred_idx += 1
            else:
                races.append("Other")
                genders.append("Male")
                groups.append("Male-Other")
                group_ids.append(GROUP_TO_IDX["Male-Other"])
                race_confs.append(0.0)
                gender_confs.append(0.0)

    df["race"] = races
    df["gender"] = genders
    df["demographic_group"] = groups
    df["group_id"] = group_ids
    df["race_confidence"] = race_confs
    df["gender_confidence"] = gender_confs

    # Save annotated CSV
    out_csv = faces_csv.replace(".csv", "_annotated.csv")
    df.to_csv(out_csv, index=False)

    # Print distribution
    print(f"\nDemographic distribution:")
    group_counts = df["demographic_group"].value_counts()
    for group_name in DEMOGRAPHIC_GROUPS:
        count = group_counts.get(group_name, 0)
        pct = count / len(df) * 100
        print(f"  {group_name:20s}: {count:6d} ({pct:5.1f}%)")

    print(f"\nSaved annotated manifest to: {out_csv}")
    return df


def annotate_all_faces(model_path=None, batch_size=32):
    """Annotate faces from all dataset manifests."""
    print("=" * 60)
    print("DEMOGRAPHIC ANNOTATION PIPELINE")
    print("=" * 60)

    all_dfs = []
    for manifest_name in ["ff_faces.csv", "celebdf_faces.csv", "dfd_faces.csv"]:
        manifest_path = os.path.join(FACES_DIR, manifest_name)
        if os.path.exists(manifest_path):
            print(f"\nAnnotating: {manifest_name}")
            df = annotate_faces_from_manifest(manifest_path,
                                               model_path=model_path,
                                               batch_size=batch_size)
            all_dfs.append(df)
        else:
            print(f"[WARN] Manifest not found: {manifest_path}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(FACES_DIR, "all_faces_annotated.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined annotated manifest: {combined_path}")
        print(f"Total annotated faces: {len(combined)}")
        return combined

    return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Annotate faces with demographics")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to FairFace model weights")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    annotate_all_faces(model_path=args.model_path)
