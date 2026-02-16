"""
PyTorch Dataset and DataLoader for deepfake detection with demographic groups.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from collections import defaultdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, NUM_GROUPS


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake detection with demographic group information.

    Returns (image_tensor, label, group_id) tuples.
    """

    def __init__(self, manifest_csv, split="train", transform=None):
        """
        Args:
            manifest_csv: Path to annotated CSV with face_path, label, group_id
            split: 'train', 'val', or 'test' — controls augmentation
            transform: Optional custom transform (overrides defaults)
        """
        self.df = pd.read_csv(manifest_csv)
        self.split = split

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.1, hue=0.05),
                transforms.RandomGrayscale(p=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        # Encode labels
        self.df["label_int"] = (self.df["label"] == "FAKE").astype(int)

        # Ensure group_id column
        if "group_id" not in self.df.columns:
            self.df["group_id"] = 0

        print(f"[Dataset] Loaded {len(self.df)} samples for {split}")
        print(f"  REAL: {(self.df['label_int']==0).sum()}, "
              f"FAKE: {(self.df['label_int']==1).sum()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        face_path = row["face_path"]
        label = row["label_int"]
        group_id = row["group_id"]

        try:
            image = Image.open(face_path).convert("RGB")
        except Exception:
            # Return a random valid index if image fails to load
            return self.__getitem__(np.random.randint(len(self.df)))

        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32)
        group_id = torch.tensor(group_id, dtype=torch.long)

        return image, label, group_id

    def get_group_distribution(self):
        """Return a dict of group_id -> count."""
        return self.df["group_id"].value_counts().to_dict()

    def get_sample_weights(self):
        """
        Compute per-sample weights for balanced sampling.
        Weights are inverse of group frequency to balance demographics.
        """
        group_counts = self.df["group_id"].value_counts()
        total = len(self.df)
        weights = []
        for _, row in self.df.iterrows():
            gid = row["group_id"]
            w = total / (NUM_GROUPS * group_counts.get(gid, 1))
            weights.append(w)
        return torch.DoubleTensor(weights)


class FairBatchSampler(Sampler):
    """
    Custom sampler that ensures each batch has representation from
    multiple demographic groups for better fairness loss computation.
    """

    def __init__(self, dataset, batch_size=BATCH_SIZE, num_groups=NUM_GROUPS):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_groups = num_groups

        # Build index lists per group
        self.group_indices = defaultdict(list)
        for idx, row in dataset.df.iterrows():
            gid = row["group_id"]
            self.group_indices[gid].append(idx)

        # Shuffle within each group
        for gid in self.group_indices:
            np.random.shuffle(self.group_indices[gid])

        self.num_samples = len(dataset)

    def __iter__(self):
        # Create balanced batches: sample proportionally from each group
        group_iters = {}
        for gid, indices in self.group_indices.items():
            np.random.shuffle(indices)
            group_iters[gid] = iter(indices)

        active_groups = list(group_iters.keys())
        samples_per_group = max(1, self.batch_size // len(active_groups))

        batch = []
        exhausted = set()

        while len(exhausted) < len(active_groups):
            for gid in active_groups:
                if gid in exhausted:
                    continue
                for _ in range(samples_per_group):
                    try:
                        idx = next(group_iters[gid])
                        batch.append(idx)
                    except StopIteration:
                        exhausted.add(gid)
                        break

                    if len(batch) >= self.batch_size:
                        yield from batch
                        batch = []

        # Yield remaining
        if batch:
            yield from batch

    def __len__(self):
        return self.num_samples


def create_dataloaders(train_csv, val_csv, test_csv=None,
                       batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                       fair_sampling=True):
    """
    Create training, validation, and optionally test DataLoaders.

    Args:
        train_csv: Path to training split CSV
        val_csv: Path to validation split CSV
        test_csv: Optional path to test split CSV
        batch_size: Batch size
        num_workers: Number of data loading workers
        fair_sampling: Whether to use FairBatchSampler for training

    Returns:
        Dict of DataLoaders: {'train': ..., 'val': ..., 'test': ...}
    """
    train_dataset = DeepfakeDataset(train_csv, split="train")
    val_dataset = DeepfakeDataset(val_csv, split="val")

    if fair_sampling:
        train_sampler = FairBatchSampler(train_dataset, batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            drop_last=True,
        )
    else:
        # Use weighted random sampling for class balance
        sample_weights = train_dataset.get_sample_weights()
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
    )

    loaders = {"train": train_loader, "val": val_loader}

    if test_csv and os.path.exists(test_csv):
        test_dataset = DeepfakeDataset(test_csv, split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
        )
        loaders["test"] = test_loader

    return loaders
